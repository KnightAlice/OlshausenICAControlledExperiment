#!/usr/bin/env bash
set -e

# Example grid searches for three transforms
# Edit params as needed.

python GridSearch.py --transform contrast --params "0, 0.5, 1, 1.5, 5, 7.5, 10" --output-dir "./outputs/contrast"  &
python GridSearch.py --transform noise --params "0,0.1,0.2,0.5,1.0, 5" --output-dir "./outputs/noise"  &
python GridSearch.py --transform blur --params "0,1,2,3,4" --output-dir "./outputs/blur"  &

wait
