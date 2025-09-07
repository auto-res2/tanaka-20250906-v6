[UPDATED]
Key fixes & improvements:
1. Updated research directory paths to `.research/iteration36` to comply with the new mandatory location rule.
2. Added optional `device` parameter to `evaluate_fid` and routed through `DiffusionTrainer.train` so that evaluation works on both CPU-only and CUDA machines.
3. Guarded CUDA–specific calls with `torch.cuda.is_available()` to avoid crashes in CPU-only CI.
4. Re-used helper `_NoOpProfiler` for CPU profiling safely.

Changed blocks are marked with "# --- fix" comments for easier diff review.