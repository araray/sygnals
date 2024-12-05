#!/usr/bin/env bash

output_dir="./outputs"
test_data_dir="./tests/data"

if [[ -d ${output_dir} ]]; then
    rm -rf ${output_dir}/
fi

mkdir -p ${output_dir} plugins

echo "Analyzing test_signal.csv..."
sygnals analyze ${test_data_dir}/test_signal.csv --output tabulated

echo "Applying FFT..."
sygnals transform ${test_data_dir}/test_signal.csv --fft --output ${output_dir}/fft_result.csv

echo "Applying Low-Pass Filter..."
sygnals filter ${test_data_dir}/test_signal.csv --low-pass 15 --output ${output_dir}/low_passed.csv

echo "Applying High-Pass Filter..."
sygnals filter ${test_data_dir}/test_signal.csv --high-pass 5 --output ${output_dir}/high_passed.csv

echo "Analyzing test_audio.wav..."
sygnals analyze ${test_data_dir}/test_audio.wav --output json

echo "Applying Time-Stretch Effect..."
sygnals audio effect ${test_data_dir}/test_audio.wav --effect stretch --factor 1.5 --output ${output_dir}/stretched_audio.wav

echo "Applying Pitch-Shift Effect..."
sygnals audio effect ${test_data_dir}/test_audio.wav --effect pitch-shift --factor 2 --output ${output_dir}/pitch_shifted_audio.wav

echo "Visualizing FFT..."
sygnals visualize ${test_data_dir}/test_signal.csv --type fft --output ${output_dir}/fft_plot.png

echo "Visualizing Spectrogram..."
sygnals visualize ${test_data_dir}/test_audio.wav --type spectrogram --output ${output_dir}/spectrogram.png

echo "Listing Plugins..."
sygnals plugin --list

echo "Applying Plugin..."
sygnals plugin amplify ${test_data_dir}/test_signal.csv --output ${output_dir}/amplified_signal.csv

echo "All tests completed!"

