import os
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from pathlib import Path
import torchaudio
from tqdm import tqdm
from typing import List
from ..utils.tpu import setup_tpu, get_tpu_rank, get_tpu_world_size, get_num_tpu_cores
from .inference import enhance, denoise


def process_batch_tpu(paths: List[Path], args, device):
    """Process a batch of files on TPU"""
    results = []
    for path in paths:
        out_path = args.out_dir / path.relative_to(args.in_dir)
        if args.parallel_mode and out_path.exists():
            continue

        dwav, sr = torchaudio.load(path)
        dwav = dwav.mean(0)

        if args.denoise_only:
            hwav, sr = denoise(
                dwav=dwav,
                sr=sr,
                device=device,
                run_dir=args.run_dir
            )
        else:
            hwav, sr = enhance(
                dwav=dwav,
                sr=sr,
                device=device,
                nfe=args.nfe,
                solver=args.solver,
                lambd=args.lambd,
                tau=args.tau,
                run_dir=args.run_dir
            )

        results.append((out_path, hwav, sr))
    return results


def _mp_fn(rank, args):
    """Main TPU process function"""
    os.environ['TPU_VISIBLE_DEVICES'] = str(rank)
    device = setup_tpu()
    world_size = get_tpu_world_size()

    # Get all audio paths
    paths = sorted(args.in_dir.glob(f"**/*{args.suffix}"))
    if len(paths) == 0:
        if rank == 0:
            print(f"No {args.suffix} files found in: {args.in_dir}")
        return

    # Split paths across TPU cores
    chunk_size = len(paths) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < world_size - 1 else len(paths)
    paths = paths[start_idx:end_idx]

    # Process in batches
    batch_size = 16
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        results = process_batch_tpu(batch_paths, args, device)

        # Save results
        for out_path, hwav, sr in results:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            xm.save(hwav[None], out_path)

        # Sync TPU cores
        xm.mark_step()

    if rank == 0:
        print(f"🌟 Enhancement done! {len(paths)} files processed on {world_size} TPU cores")


def main_tpu():
    """TPU-enabled main function"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # [Reference argument setup from __main__.py]
    args = parser.parse_args()

    # Get number of available TPU cores
    num_cores = get_num_tpu_cores()
    if num_cores < 1:
        raise RuntimeError("No TPU cores detected!")

    print(f"Launching distributed processing on {num_cores} TPU cores")

    # Launch TPU processes using detected core count
    xmp.spawn(_mp_fn, args=(args,), nprocs=num_cores)
