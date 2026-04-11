#!/bin/bash
# Helper script to run commands with the virtual environment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Run the command passed as arguments
"$@"
