# self for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py self/train_gpt2.py

from dataclasses import dataclass
import math
from torch.distributed import is_nccl_available, is_gloo_available
from torch.distributed.fsdp import ShardingStrategy
from torch import manual_seed
import os
import pickle
from logging_handler import get_logger

logger = get_logger()

@dataclass
class TrainConfig:
    # Eval only flag
    eval_only:bool = False

    # data
    dataset_dir: str = "data"
    dataset:str = 'openwebtext' # in the dataset_dir
    max_dataset_len: int = math.inf
    
    # Hugging Face pre-trained model model
    hf_model: str = None

    # logs
    log_dir: str = "logs"

    # these make the total batch size be ~0.5M
    # 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
    batch_size:int = 12 
    block_size:int = 1024

    # model
    n_layer:int = 12
    n_head:int = 12
    n_embd:int = 768
    dropout:float = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias:bool = False # do we use bias inside LayerNorm and Linear layers?

    # adamw optimizer
    learning_rate:float = 6e-4 # max learning rate
    weight_decay:float = 1e-1
    betas = (0.9, 0.999)

    vocab_size:int  = 50257 # default vocab size for GPT2
    best_val_loss:float = 1e9

    run_validation:bool = True
    device_type:str = "cuda" # "xla", "cpu",
    
    rank:int = 0
    local_rank:int = 0
    world_size:int = 1
    fsdp:bool = False
    
    seed:int = 42

    activation_checkpoint: bool = False

    mixed_precision: bool=True
    use_fp16: bool=False
    limit_all_gathers: bool=True
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    epochs: int = 3
    run_validation: bool = True
    checkpoint_dir: str = "checkpoints"
    save_ckpt:bool = True
    step_lr_gamma: float = 1e-1
    cache_dir: str = "cache"

    def __post_init__(self):
        assert self.batch_size > 0
        assert self.dropout >= 0.0 and self.dropout < 1.0
        assert self.device_type in [ "cuda", "xla", "cpu"]
        assert self.hf_model is None or \
            self.hf_model in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        manual_seed(self.seed)
        
        self.rank = int(os.getenv('RANK', 0))
        self.local_rank = int(os.getenv('LOCAL_RANK', 0))
        self.world_size = int(os.getenv('WORLD_SIZE', 1))
        self.fsdp = self.world_size > 1
        self.master_process = self.rank == 0

        self.seed = 42 + self.rank
        self.tokens_per_iter = self.world_size * self.batch_size * self.block_size

        self.dist_backend = 'nccl' if is_nccl_available() else "gloo" if is_gloo_available() else None

        self.data_dir = os.path.join(self.dataset_dir, self.dataset)
        self.train_data = os.path.join(self.data_dir, "train.bin")
        self.val_data = os.path.join(self.data_dir, "val.bin")

        self.model_name = "gpt2" if self.hf_model is None else self.hf_model
        self.optimizer_name = "adamw"

        self.log_dir = os.path.join(self.log_dir, self.model_name)
        self.cache_dir = os.path.join(self.cache_dir, self.model_name)
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_name)
        
        # attempt to derive vocab_size from the dataset
        meta_path = os.path.join(self.data_dir, 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.vocab_size = meta['vocab_size']
            logger.info(f"found vocab_size = {self.vocab_size} (inside {meta_path})")

        os.makedirs(self.log_dir, exist_ok=True)

        if self.device_type == "xla":
            os.makedirs(self.cache_dir, exist_ok=True)