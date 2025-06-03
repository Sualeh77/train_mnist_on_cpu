#!/bin/bash

# Exit on error
set -e

echo "Formatting Python files with Black..."

# Format all Python files in the project
poetry run black .

echo "Formatting complete!" 