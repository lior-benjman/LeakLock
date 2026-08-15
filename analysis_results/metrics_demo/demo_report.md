# LeakLock Metrics Demo Report

This is a **demo-only** report with illustrative numbers. It shows the shape of the metrics output for teammates before running the full evaluation.

## Report Slices

- `synthetic_only`: images marked as synthetic/generated
- `real_only`: images marked as real-world captures
- `all_images`: synthetic and real images combined

The metric definitions, thresholds, IoU matching, and risk targets are identical across all three slices.

## Summary Metrics

| slice | images | objects | predicted | image risk MAE | risk-band accuracy | high-risk precision | high-risk recall | unsupported rate | OCR success | age success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic_only | 240 | 318 | 330 | 7.8 | 0.88 | 0.91 | 0.94 | 0.01 | 0.93 | 0.96 |
| real_only | 160 | 205 | 196 | 13.4 | 0.79 | 0.83 | 0.76 | 0.06 | 0.78 | 0.90 |
| all_images | 400 | 523 | 526 | 10.1 | 0.84 | 0.88 | 0.86 | 0.03 | 0.87 | 0.94 |

## How To Read This

- Higher `precision`, `recall`, `F1`, `risk-band accuracy`, `OCR success`, and `age success` are better.
- Lower `image risk MAE`, `object risk MAE`, and `unsupported rate` are better.
- `high-risk recall` is especially important for LeakLock because missing a risky image is worse than over-warning.
- If `synthetic_only` is much better than `real_only`, the model probably needs more real-world training examples or less synthetic bias.

## Example Takeaways

1. Synthetic images perform best, with strong high-risk recall and low unsupported rate.
2. Real images are harder: OCR success and high-risk recall drop, which suggests real-world blur, lighting, occlusion, and document variation are affecting the pipeline.
3. Combined metrics are useful for the overall headline, but the split metrics explain where the system is weak.

## Files

- `metrics_summary_demo.csv`: one row per report slice
- `metrics_per_class_demo.csv`: per-class precision, recall, and F1 per report slice
