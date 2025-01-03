import functools
import torch.nn as nn

try:
    from torch_xla.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        size_based_auto_wrap_policy,
    )
except ImportError:
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        size_based_auto_wrap_policy,
    )

def get_size_policy(min_params=1e8) -> functools.partial:
    num_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=min_params
    )
    return num_wrap_policy


def get_transformer_wrapper(wrap_class:nn.Module) -> functools.partial:
    gpt_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls = {
            wrap_class
        }
    )

    return gpt_auto_wrap_policy
