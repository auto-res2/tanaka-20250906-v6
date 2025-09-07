[UPDATED]
Changes:
1. Switched all hard-coded iteration34 paths to iteration36.
2. Added safe device handling – no more unconditional CUDA calls.  Uses the same device as the model (passed from caller) and falls back cleanly on CPU.
3. Added context-manager logic to disable autocast on CPU while keeping AMP on CUDA.
4. Signature changed to `evaluate_fid(model, val_loader, scheduler, cfg, device)`.
