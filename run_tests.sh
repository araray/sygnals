#!/usr/bin/env bash

output_dir="./outputs"
test_data_dir="./tests/data"

if [[ -d ${output_dir} ]]; then
    rm -rf ${output_dir}/
fi

mkdir -p ${output_dir} plugins

function check_output_file() {
    if [[ ! -f $1 ]]; then
        echo "Error: Output file '$1' not found!"
    fi
}

tests_count=0
tests_failed=0

echo "Analyzing test_signal.csv..."
let tests_count++
sygnals analyze ${test_data_dir}/test_signal.csv --output tabulated
rc=$?
if [[ $rc -ne 0 ]]; then
    let tests_failed++
fi

echo "Applying FFT..."
output_file="${output_dir}/fft_result.csv"
let tests_count++
sygnals transform ${test_data_dir}/test_signal.csv --fft --output ${output_file}
rc=$?
if [[ $rc -ne 0 ]]; then
    let tests_failed++
fi
check_output_file ${output_file}

echo "Applying Low-Pass Filter..."
output_file="${output_dir}/low_passed.csv"
let tests_count++
sygnals filter ${test_data_dir}/test_signal.csv --low-pass 15 --output ${output_file}
rc=$?
if [[ $rc -ne 0 ]]; then
    let tests_failed++
fi
check_output_file ${output_file}

echo "Applying High-Pass Filter..."
output_file="${output_dir}/high_passed.csv"
let tests_count++
sygnals filter ${test_data_dir}/test_signal.csv --high-pass 5 --output ${output_file}
rc=$?
if [[ $rc -ne 0 ]]; then
    let tests_failed++
fi
check_output_file ${output_file}

echo "Analyzing test_audio.wav..."
let tests_count++
sygnals analyze ${test_data_dir}/test_audio.wav --output json
rc=$?
if [[ $rc -ne 0 ]]; then
    let tests_failed++
fi

echo "Applying Time-Stretch Effect..."
output_file="${output_dir}/stretched_audio.wav"
let tests_count++
sygnals audio effect ${test_data_dir}/test_audio.wav --effect stretch --factor 1.5 --output ${output_file}
rc=$?
if [[ $rc -ne 0 ]]; then
    let tests_failed++
fi
check_output_file ${output_file}

echo "Applying Pitch-Shift Effect..."
output_file="${output_dir}/pitch_shifted_audio.wav"
let tests_count++
sygnals audio effect ${test_data_dir}/test_audio.wav --effect pitch-shift --factor 2 --output ${output_file}
rc=$?
if [[ $rc -ne 0 ]]; then
    let tests_failed++
fi
check_output_file ${output_file}

echo "Visualizing FFT..."
output_file="${output_dir}/fft_plot.png"
let tests_count++
sygnals visualize ${test_data_dir}/test_signal.csv --type fft --output ${output_file}
rc=$?
if [[ $rc -ne 0 ]]; then
    let tests_failed++
fi
check_output_file ${output_file}

echo "Visualizing Spectrogram..."
output_file="${output_dir}/spectrogram.png"
let tests_count++
sygnals visualize ${test_data_dir}/test_audio.wav --type spectrogram --output ${output_file}
rc=$?
if [[ $rc -ne 0 ]]; then
    let tests_failed++
fi
check_output_file ${output_file}

echo "Listing Plugins..."
let tests_count++
sygnals plugin --list
rc=$?
if [[ $rc -ne 0 ]]; then
    let tests_failed++
fi

echo "Applying Plugin..."
output_file="${output_dir}/amplified_signal.csv"
let tests_count++
sygnals plugin amplify ${test_data_dir}/test_signal.csv --output ${output_file}
rc=$?
if [[ $rc -ne 0 ]]; then
    let tests_failed++
fi
check_output_file ${output_file}

echo "All tests completed!"
echo "Tests count: ${tests_count}"
echo "Tests failed: ${tests_failed}"
exit ${tests_failed}
