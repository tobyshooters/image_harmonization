# --------------------------------------
# Run locally:
# --------------------------------------
 python3 scripts/predict_for_dir.py          \
     hrnet18_idih256                         \
     checkpoints/hrnet18_idih256.pth         \
     --images HAdobe5k/composite_images       \
     --masks HAdobe5k/masks              \
     --results-path HAdobe5k/results     \
     --gpu -1                                \
     --resize 256                            \
     --original-size                         \
# --------------------------------------
