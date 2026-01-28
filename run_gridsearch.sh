#!/usr/bin/env bash
set -e

# Example grid searches for three transforms
# Edit params as needed.

python GridSearch.py --transform contrast --params "0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2" --output-dir "./outputs/contrast" --device cuda &
python GridSearch.py --transform noise --params "0,0.1,0.2,0.5,1.0, 5" --output-dir "./outputs/noise" --device cuda &
python GridSearch.py --transform blur --params "0,1,2,3,4" --output-dir "./outputs/blur" --device cuda &

wait
