import os
import socket
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


def get_free_port():
    """Get an available port."""
    s = socket.socket()
    s.bind(('', 0))        # Bind to a free port provided by the host.
    location = s.getsockname()
    s.close()
    return location[1]


def setup_tpu_port():
    """Configure TPU port settings"""
    if 'TPU_METRICS_PORT' not in os.environ:
        # Pick a random port to avoid conflicts
        port = 8471 + int(os.environ.get('LOCAL_RANK', '0'))
        os.environ['TPU_METRICS_PORT'] = str(port)

    # Configure XLA port if not set
    if 'XRT_TPU_CONFIG' not in os.environ:
        # Use a fixed port for simplicity, adjust if needed for multi-host setup
        # It's important this port doesn't conflict with TPU_METRICS_PORT
        xla_port = 9666  # Using a different default port
        tpu_config = f"tpu_worker;0;localhost:{xla_port}"
        os.environ['XRT_TPU_CONFIG'] = tpu_config


def setup_tpu():
    """Initialize TPU device and return device object"""
    setup_tpu_port()
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
