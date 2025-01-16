import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl


def get_num_tpu_cores():
    """Detect number of available TPU cores"""
    try:
        # Try to get from environment first
        if 'TPU_NUM_DEVICES' in os.environ:
            return int(os.environ['TPU_NUM_DEVICES'])

        # Fallback to XLA runtime detection
        devices = torch_xla._XLAC._xla_get_devices()
        return len(devices)
    except:
        # If detection fails, return 1 to allow single-core execution
        return 1


def setup_tpu():
    """Initialize TPU device and return device object"""
    device = xm.xla_device()
    return device


def get_tpu_rank():
    """Get current TPU core rank"""
    return xm.get_ordinal()


def get_tpu_world_size():
    """Get total number of TPU cores"""
    return xm.xrt_world_size()


def create_tpu_dataloader(dataset, batch_size, shuffle=True):
    """Create TPU-optimized dataloader"""
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=get_tpu_world_size(),
        rank=get_tpu_rank(),
        shuffle=shuffle
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        drop_last=False
    )

    return dataloader, sampler
