#!/usr/bin/env bash

sygnals_test="${1:-.}"

source venv/bin/activate

{
    python -m pytest --disable-warnings -l ${sygnals_test} 2>&1
} | tee -a /av/outputs/sygnals_tests_run_$(date +'%s').txt

deactivate
