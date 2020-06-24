# --------------------------------------
# Run locally:
# --------------------------------------
 python3 scripts/predict_for_dir.py          \
     hrnet18_idih256                         \
     checkpoints/hrnet18_idih256.pth         \
     --images test_data/sample/comps/        \
     --masks test_data/sample/masks/         \
     --results-path test_data/sample/results \
     --gpu -1                                \
     --resize 256                            \
     --original-size                         \
# --------------------------------------
