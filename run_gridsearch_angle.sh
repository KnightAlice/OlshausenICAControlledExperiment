#!/usr/bin/env bash
set -e

# Example grid searches for three transforms
# Edit params as needed.

#python GridSearch_angle_prediction.py --transform contrast --params "0,0.25,0.5,0.75,1,1.25,1.5,2" --output-dir "./outputs_angle/contrast" --device cuda --n-samples 1000 --angle-jitter-deg 10 --gabor-sigma 2.0 --gabor-freq 0.1 --gabor-phase 0.0 &
#python GridSearch_angle_prediction.py --transform contrast --params "0,0.001,0.002,0.003,0.004,0.005,0.0075,0.01,0.015,0.02,0.03,0.04" --output-dir "./outputs_angle/contrast" --device cuda --n-samples 1000 --angle-jitter-deg 10 --gabor-sigma 2.0 --gabor-freq 0.1 --gabor-phase 0.0 &
#python GridSearch_angle_prediction.py --n-units 180 --num-repeats 10 --transform contrast --params "0,0.001,0.002,0.003,0.004,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.1" --output-dir "./outputs_angle/contrast" --device cuda --n-samples 1000 --angle-jitter-deg 10 --gabor-sigma 2.0 --gabor-freq 0.1 --gabor-phase 0.0 &

#Best
python GridSearch_angle_prediction.py --n-units 180 --num-repeats 5 --transform contrast --params "0,0.001,0.002,0.003,0.004,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.1" --output-dir "./outputs_angle_test/contrast" --device cuda --n-samples 1000 --angle-jitter-deg 10 --gabor-sigma 2.0 --gabor-freq 0.1 --gabor-phase 0.0 &

python GridSearch_angle_prediction.py --n-units 180 --num-repeats 5 --transform noise --params "0,0.1,0.2,0.5,1.0,1.25,1.5" --output-dir "./outputs_angle_test/noise" --device cuda --n-samples 1000 --angle-jitter-deg 10 --gabor-sigma 2.0 --gabor-freq 0.1 --gabor-phase 0.0 &
python GridSearch_angle_prediction.py --n-units 180 --num-repeats 5 --transform blur --params "0,1,2,3,4" --output-dir "./outputs_angle_test/blur" --device cuda --n-samples 1000 --angle-jitter-deg 10 --gabor-sigma 2.0 --gabor-freq 0.1 --gabor-phase 0.0 &

# python GridSearch_angle_prediction.py --n-units 180 --num-repeats 3 --blur-mode "gaussian" --transform contrast --params "0,0.001,0.002,0.003,0.004,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.1" --output-dir "./outputs_angle_gauss_test/contrast" --device cuda --n-samples 1000 --angle-jitter-deg 10 --gabor-sigma 2.0 --gabor-freq 0.1 --gabor-phase 0.0 &

# python GridSearch_angle_prediction.py --n-units 180 --num-repeats 3 --blur-mode "gaussian" --transform noise --params "0,0.1,0.2,0.5,1.0,1.5" --output-dir "./outputs_angle_gauss_test/noise" --device cuda --n-samples 1000 --angle-jitter-deg 10 --gabor-sigma 2.0 --gabor-freq 0.1 --gabor-phase 0.0 &
# python GridSearch_angle_prediction.py --n-units 180 --num-repeats 3 --blur-mode "gaussian" --transform blur --params "0,1,2,3,4" --output-dir "./outputs_angle_gauss_test/blur" --device cuda --n-samples 1000 --angle-jitter-deg 10 --gabor-sigma 2.0 --gabor-freq 0.1 --gabor-phase 0.0 &


wait

#Possible Cause:
#1. SSN is not trained on the new patches dataset.
#2. SSN is sensitive to pixel values, so it is prone to unnormalized distortions.