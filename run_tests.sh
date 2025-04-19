#!/usr/bin/env bash

source venv/bin/activate

{
    python -m pytest . 2>&1
} | tee -a /av/outputs/sygnals_run_$(date +'%s').txt

deactivate
