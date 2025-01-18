import os
import torch
import argparse
from pathlib import Path
import torchaudio
from tqdm import tqdm
from typing import List

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    xla_available = True
except ImportError:
    print("torch_xla not found. Running on CPU/GPU.")
    xla_available = False

from resemble_enhance.enhancer.inference import enhance, denoise


def process_batch(paths: List[Path], args, device):
    """Process a batch of files on the specified device"""
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
    """Main process function for multi-processing"""
    if xla_available:
        device = xm.xla_device()
        world_size = xm.xrt_world_size()
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        world_size = 1  # Single process

    # Get all audio paths
    paths = sorted(args.in_dir.glob(f"**/*{args.suffix}"))
    if len(paths) == 0:
        if rank == 0:
            print(f"No {args.suffix} files found in: {args.in_dir}")
        return

    # Split paths across processes
    chunk_size = len(paths) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < world_size - 1 else len(paths)
    paths = paths[start_idx:end_idx]

    # Process in batches
    batch_size = 16
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        results = process_batch(batch_paths, args, device)

        # Save results
        for out_path, hwav, sr in results:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Use torch.save for broader compatibility
            torch.save((hwav, sr), out_path)

        # Sync processes if using XLA
        if xla_available:
            xm.mark_step()

    if rank == 0:
        print(f"🌟 Enhancement done! {len(paths)} files processed on {'TPU' if xla_available else 'CPU/GPU'}")
        if xla_available:
            print(f"  {world_size} TPU cores were used.")


def main_tpu():
    """Main function to handle argument parsing and process launching"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('in_dir', type=Path, help='Input directory')
    parser.add_argument('out_dir', type=Path, help='Output directory')
    parser.add_argument('--suffix', type=str,
                        default=".wav", help='Audio file suffix')
    parser.add_argument('--parallel_mode', action='store_true',
                        help='Skip existing files in output directory')
    parser.add_argument('--denoise_only', action='store_true',
                        help='Run only the denoiser')
    parser.add_argument('--nfe', type=int, default=32,
                        help='Number of function evaluations for DiffWave')
    parser.add_argument('--solver', type=str,
                        default='midpoint', help='Solver type for DiffWave')
    parser.add_argument('--lambd', type=float, default=0.99,
                        help='Lambda parameter for DiffWave')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Tau parameter for DiffWave')
    parser.add_argument('--run_dir', type=Path, default='./run',
                        help='Directory to save intermediate files')

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if xla_available:
        num_cores = int(os.environ.get('XRT_NUM_DEVICES', xm.xrt_world_size()))
        print(f"Launching distributed processing on {num_cores} TPU cores")
        xmp.spawn(_mp_fn, args=(args,), nprocs=num_cores, start_method='fork')
    else:
        print("Running on CPU/GPU.")
        _mp_fn(0, args)  # Run directly on single process


if __name__ == "__main__":
    main_tpu()
