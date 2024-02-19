"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

from ast import literal_eval
import os
import time
import sys

import torch
from torch import distributed as dist
import tqdm

from model import GPT2Config, GPT2
from gpt_dataset import GptDataset
from dataclasses import replace
from train_config import TrainConfig
import policies
from torch.optim.lr_scheduler import StepLR

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, DataLoader

from checkpoint_handler import CheckpointHandler

from model import GPT2TransformerBlock

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend
    import torch_xla.runtime as xr
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
    from torch_xla.distributed.fsdp import checkpoint_module
except ImportError:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from activation_checkpoint_handler import checkpoint_activation

from torch.utils.tensorboard import SummaryWriter

from logging_handler import get_logger
logger = get_logger()

g_gigabyte:int = 1024**3

def get_sys_kwargs(train_config:TrainConfig):
    kwargs = dict()
    for arg in sys.argv[1:]:
        if '='  in arg:
            # assume it's a --key=value argument
            assert arg.startswith('--')
            key, value = arg.split('=')
            key = key[2:]
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(value)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = value
                
            if hasattr(train_config, key):
                attr_type = type(getattr(train_config, key))
                if type(attempt) != attr_type:
                    logger.warning(f"key:{key}, value:{attempt}, key type{attr_type}, value type {type(attempt)}")
            kwargs[key] = attempt
            
    return kwargs

class TrainFSDP:

    def __init__(self, cfg:TrainConfig):
        self.cfg = cfg
        self.__ckpt_handler = CheckpointHandler(cfg)
        if self.cfg.rank == 0:
            self.__summary_writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, cfg.device_type))

        if self.cfg.device_type == "xla":
            compiler_cache_path = self.cfg.cache_dir
            os.makedirs(compiler_cache_path, exist_ok=True)
            xr.initialize_cache(compiler_cache_path, readonly=False)

    def __get_policies(self, wrap_cls:torch.nn.Module, bf16_supported:bool=False):
        mixed_precision_policy = None
        wrapping_policy = None

        # mixed precision -----
        if self.cfg.mixed_precision:
            if bf16_supported and not self.cfg.use_fp16:
                mixed_precision_policy = policies.bfSixteen
                if self.cfg.rank == 0:
                    logger.info(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
            elif self.cfg.use_fp16:
                mixed_precision_policy = policies.fpSixteen
                if self.cfg.rank == 0:
                    logger.info(f"FP16 enabled. ")
            else:
                # mixed_precision_policy = policies.fpSixteen
                logger.info(f"bFloat16 support not present. Will use FP32, and not mixed precision")

        
        wrapping_policy = policies.get_transformer_wrapper(wrap_cls)

        return mixed_precision_policy, wrapping_policy


    def __train_step(self, model:GPT2, 
                optimizer:torch.optim.Optimizer, 
                fsdp_loss:torch.Tensor,
                X: torch.Tensor,
                Y: torch.Tensor):
        
        optimizer.zero_grad()
        _, loss = model(X, Y)
        loss.backward()

        optimizer.step()
        
        with torch.no_grad():
            fsdp_loss[0] = torch.add(fsdp_loss[0], loss)
            fsdp_loss[1] = torch.add(fsdp_loss[1], self.cfg.batch_size)


    def __train_epoch(self, model:GPT2, 
            train_loader:DataLoader, 
            optimizer:torch.optim.AdamW, 
            epoch:int, 
            device:torch.device):

        model.train()
        fsdp_loss = torch.zeros(2).to(device=device)
    
        if self.cfg.rank==0:
            inner_pbar = tqdm.tqdm(range(len(train_loader)), 
                                colour="blue", desc=f"Training Epoch {epoch}:")

        iters = 0
        for X,Y in train_loader:
            X, Y = X.to(device=device), Y.to(device=device)
        
            self.__train_step(model=model, 
                    optimizer=optimizer, 
                    fsdp_loss=fsdp_loss,
                    X=X, Y=Y)
            if self.cfg.rank==0:
                inner_pbar.update(1)

            iters += 1

        if self.cfg.fsdp:
            if self.cfg.device_type == "xla":
                xm.all_reduce(xm.REDUCE_SUM, fsdp_loss)
            else:
                dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)

        train_loss = fsdp_loss[0] / fsdp_loss[1]

        if self.cfg.rank == 0:
            inner_pbar.close()
            logger.info(f"Train Epoch: \t{epoch}, Loss: \t{train_loss:.4f}")
        return train_loss, iters

    def __validation(self, model:GPT2, val_loader:DataLoader, device:torch.device):
        model.eval()
    
        fsdp_loss = torch.zeros(2).to(device=device)
        if self.cfg.rank == 0:
            inner_pbar = tqdm.tqdm(
                range(len(val_loader)), colour="green", desc="Validation Epoch"
            )
        with torch.no_grad():
            for X,Y in val_loader:
                X, Y = X.to(device=device), Y.to(device=device)

                _, loss = model(X, Y)
                
                fsdp_loss[0] = torch.add(fsdp_loss[0], loss)
                fsdp_loss[1] = torch.add(fsdp_loss[1], self.cfg.batch_size)

                if self.cfg.rank==0:
                    inner_pbar.update(1)

                if self.cfg.device_type == "xla":
                    xm.mark_step()

        if self.cfg.fsdp:
            if self.cfg.device_type == "xla":
                xm.all_reduce(xm.REDUCE_SUM, fsdp_loss)
            else:
                dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)

        val_loss = fsdp_loss[0] / fsdp_loss[1]
        if self.cfg.rank == 0:
            inner_pbar.close()
            logger.info(f"Validation Loss: {val_loss:.4f}")
        return val_loss


    def __init_model(self, model_config: dict):
       
        if self.cfg.hf_model is None:
            gptconf = GPT2Config(**model_config)
            model = GPT2(gptconf)
        else:
            model = GPT2.from_hf_model(self.cfg.hf_model)

        return model

    def __initialize(self, model_config: dict, bf16_supported:bool, device:torch.device):
        model = self.__init_model(model_config=model_config)
        epoch = 1

        if self.cfg.fsdp:
            wrap_cls = GPT2TransformerBlock

            mixed_precision_policy, gpt2_auto_wrap_policy = \
                self.__get_policies(wrap_cls, bf16_supported=bf16_supported)
            
            # Apply FSDP wrapping to the model
            if self.cfg.device_type == "xla":
                compute_dtype= torch.bfloat16 \
                    if self.cfg.mixed_precision and bf16_supported else torch.float32 
                fp32_reduce_scatter = True \
                    if self.cfg.mixed_precision and bf16_supported else False
                
                auto_wrapper_callable = None
                if self.cfg.activation_checkpoint:
                    auto_wrapper_callable = lambda m, *args, **kwargs: \
                            FSDP(checkpoint_module(m), *args, **kwargs)
                model = FSDP(model,
                    reshard_after_forward = True,
                    execute_sharding_on_init = True,
                    optimization_barrier_in_forward = True,
                    optimization_barrier_in_backward = True,
                    mark_step_on_finalization = True,
                    auto_wrap_policy=gpt2_auto_wrap_policy,
                    compute_dtype=compute_dtype,
                    fp32_reduce_scatter=fp32_reduce_scatter,
                    auto_wrapper_callable = auto_wrapper_callable
                )
            else:
                model = FSDP(model,
                    auto_wrap_policy=gpt2_auto_wrap_policy,
                    mixed_precision=mixed_precision_policy,
                    sharding_strategy=self.cfg.sharding_strategy,
                    device_id=device,
                    limit_all_gathers=self.cfg.limit_all_gathers)
                if self.cfg.activation_checkpoint:
                    checkpoint_activation(model, wrap_cls)
                
            optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=self.cfg.learning_rate,
                                    weight_decay=self.cfg.weight_decay,
                                    betas = self.cfg.betas)

            epoch = self.__ckpt_handler.load(model, optimizer)
        else:
            model = model.to(device=device)
            optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=self.cfg.learning_rate,
                                    weight_decay=self.cfg.weight_decay,
                                    betas = self.cfg.betas)
            epoch = self.__ckpt_handler.load(model, optimizer)
            
        return model, optimizer, epoch

    def __get_device(self):
        
        if self.cfg.device_type == 'xla':
            device = xm.xla_device()
            bf16_supported = bool(os.getenv('XLA_USE_BF16', False))

            xla_rank = xm.get_ordinal()
            assert self.cfg.rank == xla_rank, f"Rank {self.cfg.rank} vs. {xla_rank}"

            xla_world_size = xm.xrt_world_size()
            assert self.cfg.world_size == xla_world_size, f"World size {self.cfg.world_size} vs. {xla_world_size}"
        elif self.cfg.device_type == 'cuda':
            device = torch.device(f"cuda:{self.cfg.local_rank}")
            bf16_supported = torch.cuda.is_bf16_supported()
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        else:
            device = torch.device("cpu")
            bf16_supported = False

        return device, bf16_supported

    def __call__(self):

        if self.cfg.fsdp:
            if self.cfg.device_type == "xla":
                logger.info(f"init process group using init method xla://")
                dist.init_process_group('xla', init_method='xla://')
            else:
                logger.info(f"init process group using backend: '{self.cfg.dist_backend}'")
                dist.init_process_group(self.cfg.dist_backend)

        device, bf16_supported = self.__get_device()

        # dataloaders
        train_dataset = GptDataset(self.cfg.train_data, 
                                   block_size=self.cfg.block_size, max_len=self.cfg.max_dataset_len)
        val_dataset = GptDataset(self.cfg.val_data, block_size=self.cfg.block_size)

        if self.cfg.fsdp:
            train_sampler=DistributedSampler(train_dataset, 
                                            num_replicas=self.cfg.world_size, 
                                            rank=self.cfg.rank, shuffle=True, 
                                            seed=0, drop_last=True)
            val_sampler=DistributedSampler(val_dataset, 
                                            num_replicas=self.cfg.world_size, 
                                            rank=self.cfg.rank, shuffle=False, 
                                            seed=0, drop_last=True)
        else:
            train_sampler = RandomSampler(train_dataset)
            val_sampler = RandomSampler(val_dataset)

        extra_kwargs = {'batch_size': self.cfg.batch_size,  'sampler': train_sampler}
        val_kwargs = {'batch_size': self.cfg.batch_size, 'sampler': val_sampler}
        addl_kwargs = {'num_workers': 1, 'prefetch_factor': 2, 'shuffle': False}
        extra_kwargs.update(addl_kwargs)
        val_kwargs.update(addl_kwargs)

        train_loader = DataLoader(train_dataset,**extra_kwargs)
        val_loader = DataLoader(val_dataset, **val_kwargs)
        
        model_config = dict(n_layer=self.cfg.n_layer, 
                            n_head=self.cfg.n_head, 
                            n_embd=self.cfg.n_embd, 
                            block_size=self.cfg.block_size,
                            bias=self.cfg.bias, 
                            vocab_size=self.cfg.vocab_size, 
                            dropout=self.cfg.dropout,
                            activation_checkpoint=self.cfg.activation_checkpoint)

        
        model, optimizer, start_epoch = self.__initialize(model_config=model_config, 
                                            bf16_supported=bf16_supported,
                                            device=device)

        scheduler = StepLR(optimizer, step_size=1, gamma=self.cfg.step_lr_gamma)
        best_val_loss = float("inf")
        curr_val_loss = float("inf")

        total_duration = 0
        total_iters = 0

        for epoch in range(start_epoch, start_epoch + self.cfg.epochs):
            t0 = time.time()
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)
            train_loss,epoch_iters = self.__train_epoch(model, train_loader, optimizer, epoch, device)
            if self.cfg.run_validation:
                curr_val_loss = self.__validation(model, val_loader, device)
            scheduler.step()
            
            if self.cfg.rank == 0:
                epoch_duration = time.time() - t0
                total_duration += epoch_duration
                total_iters += epoch_iters
                self.__summary_writer.add_scalar('Epoch/duration-secs', epoch_duration, epoch)
                self.__summary_writer.add_scalar('Loss/train', train_loss.item(), epoch)
                if self.cfg.run_validation:
                    self.__summary_writer.add_scalar('Loss/validation', curr_val_loss.item(), epoch)

                logger.info(f"Epoch: {epoch} completed in {epoch_duration} seconds")

            if curr_val_loss < best_val_loss:
                best_val_loss = curr_val_loss
                if self.cfg.rank==0:
                    logger.info(f"New validation Loss Record: {best_val_loss}, epoch: {epoch}")

                if self.cfg.save_ckpt:
                    logger.info(f"Checkpointing at epoch: {epoch}, rank: {self.cfg.rank}")
                    self.__ckpt_handler.save(model, optimizer, epoch)

        if self.cfg.rank == 0:
            logger.info(f"Total duration (secs): {total_duration: 0.3f}")
            logger.info(f"Total iterations: {total_iters}")
            
            avg_iers_per_sec = total_iters/total_duration
            logger.info(f"Average iters/sec: {avg_iers_per_sec: 0.2f}")

            avg_tokens_per_sec = (total_iters * self.cfg.tokens_per_iter)/total_duration
            logger.info(f"Average tokens/sec: {avg_tokens_per_sec: 0.2f}")
            self.__summary_writer.close()

        if self.cfg.fsdp:
            logger.info(f"Rank {self.cfg.rank}: barrier")
            dist.barrier()
            logger.info(f"Rank {self.cfg.rank}: destroy_process_group")
            dist.destroy_process_group()

if __name__ == "__main__":
    cfg = TrainConfig()
    cfg = replace(cfg, **get_sys_kwargs(train_config=cfg))
    train_fsdp = TrainFSDP(cfg=cfg)

    logger.info(cfg)
    logger.info(f"Tokens per iteration: {cfg.tokens_per_iter}")
    train_fsdp()
    logger.info(f"Rank {cfg.rank}: done.")