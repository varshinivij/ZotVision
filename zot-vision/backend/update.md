# Backend Updates

## transformer.py

### Bug Fixes
- **`load_samples()`**: Was expecting `filename,label` (comma-separated) per line but `labels.txt` has one label per line. Fixed to map line `i` to image `{i+1}.jpg`.
- **`pos_embed`**: Was initialized as `None` in `__init__` and lazily created as a new `nn.Parameter` inside `forward()`. This meant it was never registered with PyTorch — excluded from `state_dict` (weights wouldn't save/load), and not moved to device on `.to()`. Fixed by registering it as a proper `nn.Parameter` in `__init__` (pre-computed for EfficientNet-B4 @ 224×224 → 49 patches).
- **`_build_pos_embed()` removed**: Dead code after the `pos_embed` fix. Replaced with an interpolation fallback in `forward()` for non-standard input resolutions.

## train_colab.ipynb (new)

Colab notebook for training on TPU. Includes the same `transformer.py` fixes plus:
- Google Drive mount and dataset zip extraction
- TPU device setup via `torch_xla`
- `num_workers=0` for XLA compatibility
- `xm.optimizer_step()` and `xm.mark_step()` in training loop
- `xm.save()` for TPU-safe weight saving to Google Drive

## auto_labeling.py

- **Sort fix**: Images were sorted lexicographically (`1, 10, 100...`) causing label-to-image mismatches. Fixed to sort numerically by filename stem.
- **Upgraded CLIP model**: Switched from `ViT-B-32` to `ViT-L-14 (OpenAI)` for significantly better zero-shot accuracy.
- **Improved prompts**: Added more smoke-specific prompts (white, gray, light smoke variants) and additional fire prompts to improve recall.
- **Dual-condition hazard detection**: Flags hazard if mean margin > 0.03 OR any single prompt similarity > 0.25, catching subtle smoke that scores high on one prompt but dilutes the mean.
- **Margin threshold**: Raised from `0.02` to `0.03` to reduce false positives.
