# --------------------------------------
# Run locally:
# --------------------------------------
 python3 scripts/predict_for_dir.py          \
     hrnet18_idih256                         \
     checkpoints/hrnet18_idih256.pth         \
     --images HAdobe5k/composite_images      \
     --masks HAdobe5k/masks                  \
     --results-path HAdobe5k/results_$1   \
     --test-path HAdobe5k/HAdobe5k_test.txt  \
     --gpu -1                                \
     --resize $1                           \
     --original-size                         \
     --color-transfer                        \
     --num-inputs 10                         \
# --------------------------------------
