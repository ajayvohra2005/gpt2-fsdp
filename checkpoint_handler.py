import os
import torch

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import (
        StateDictType,
        ShardedStateDictConfig,
        ShardedOptimStateDictConfig
    )

from train_config import TrainConfig

from logging_handler import get_logger
logger = get_logger()

class CheckpointHandler:

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.__init_chkpt_path()

        if self.cfg.device_type == "cuda":
            self.sharded_state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
            self.sharded_optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=True)

    def __init_chkpt_path(self):
        self.__chkpt_path = os.path.join(self.cfg.checkpoint_dir,
                                         self.cfg.device_type,
                                         f"rank_{self.cfg.rank}-world_{self.cfg.world_size}.pt" )

        logger.info(f"Checkpoint path: {self.__chkpt_path}")               
        os.makedirs(os.path.dirname(self.__chkpt_path), exist_ok=True)
   
    def __load_checkpoint(self, model, optimizer):
        epoch = 1
        try:
            state_dict = torch.load(self.__chkpt_path)
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            epoch = state_dict['epoch'] + 1
        except Exception as e:
            logger.info(f"load chkpt error: {e}")
        return epoch

    def __load_fsdp_checkpoint(self, model, optimizer):
        epoch = 1
        try:
            state_dict = torch.load(self.__chkpt_path)
            with FSDP.state_dict_type(model, 
                                      StateDictType.SHARDED_STATE_DICT, 
                                      self.sharded_state_dict_config,
                                      self.sharded_optim_state_dict_config):
                model.load_state_dict(state_dict['model'])
                optim_state_dict = FSDP.optim_state_dict_to_load(model, optimizer, state_dict['optimizer'])
                optimizer.load_state_dict(optim_state_dict)
            epoch = state_dict['epoch'] + 1
        except Exception as e:
            logger.info(f"load chkpt error: {e}")
        return epoch
   
    def __save_checkpoint(self, model, optimizer, epoch):
        try:
            msd = model.state_dict()
            osd = optimizer.state_dict()
            torch.save({"model": msd, "optimizer": osd, "epoch": epoch}, self.__chkpt_path)
        except Exception as e:
            logger.warning(f"save chkpt error: {e}")

    def __save_fsdp_checkpoint(self, model, optimizer, epoch):
        try:
            with FSDP.state_dict_type(model, 
                                      StateDictType.SHARDED_STATE_DICT, 
                                      self.sharded_state_dict_config,
                                      self.sharded_optim_state_dict_config):
                model_state_dict = model.state_dict()
                optim_state_dict = FSDP.optim_state_dict(model, optimizer)
                torch.save({"model": model_state_dict, 
                            "optimizer": optim_state_dict, "epoch": epoch}, self.__chkpt_path)
        except Exception as e:
            logger.warning(f"save chkpt error: {e}")

    def __save_fsdp_xla_checkpoint(self, model, optimizer, epoch):
        try:
            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            xm.save(state_dict, self.__chkpt_path, master_only=False)
        except Exception as e:
            logger.warning(f"save chkpt error: {e}")

    def save(self, model, optimizer, epoch):
        if self.cfg.fsdp:
            if self.cfg.device_type == "xla":
                self.__save_fsdp_xla_checkpoint(model, optimizer, epoch)
            else:
                self.__save_fsdp_checkpoint(model, optimizer, epoch)
        else:
            self.__save_checkpoint(model, optimizer, epoch)

    def load(self, model, optimizer):
        if not os.path.isfile(self.__chkpt_path):
            logger.info("No checkpoint available")
            return 1
        
        if self.cfg.fsdp:
            return self.__load_fsdp_checkpoint(model, optimizer)
        else:
            return self.__load_checkpoint(model, optimizer)
