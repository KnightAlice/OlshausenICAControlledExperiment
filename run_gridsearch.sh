#!/usr/bin/env bash
set -e

# Example grid searches for three transforms
# Edit params as needed.

python GridSearch_representation_allignment.py --transform contrast --params "0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2" --output-dir "./outputs/contrast" --device cuda &
python GridSearch_representation_allignment.py --transform noise --params "0,0.1,0.2,0.5,1.0, 1.5" --output-dir "./outputs/noise" --device cuda &
python GridSearch_representation_allignment.py --transform blur --params "0,1,2,3,4" --output-dir "./outputs/blur" --device cuda &

wait

#Possible Cause:
#1. SSN is not trained on the new patches dataset.
#2. SSN is sensitive to pixel values, so it is prone to unnormalized distortions.