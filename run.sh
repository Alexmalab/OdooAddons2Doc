#!/bin/bash

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    echo "Please set it using: export OPENAI_API_KEY=your_api_key"
    exit 1
fi

# Default values
ADDONS_PATH=""
OUTPUT_PATH="./output"
MODEL="gpt-4o"
MAX_WORKERS=4
MAX_FILE_TOKENS=3500
BATCH_SIZE=10
RATE_LIMIT=10.0
BASE_URL=""
MERGE_ONLY=0

# Set HTTP proxy if needed (this will be used by OpenAI library automatically)
# export HTTP_PROXY=127.0.0.1:7890
# export HTTPS_PROXY=127.0.0.1:7890

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --addons-path)
            ADDONS_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --max-file-tokens)
            MAX_FILE_TOKENS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --rate-limit)
            RATE_LIMIT="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --merge-only)
            MERGE_ONLY=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if addons path is provided
if [ -z "$ADDONS_PATH" ] && [ $MERGE_ONLY -eq 0 ]; then
    echo "Error: --addons-path is required."
    echo "Usage: ./run.sh --addons-path /path/to/addons [--output-path ./output] [--model gpt-4o] [--max-workers 4]"
    echo "                [--max-file-tokens 3500] [--batch-size 10] [--rate-limit 10.0] [--base-url URL] [--merge-only]"
    echo
    echo "Note: HTTP proxy should be set as environment variables:"
    echo "      export HTTP_PROXY=127.0.0.1:7890"
    echo "      export HTTPS_PROXY=127.0.0.1:7890"
    exit 1
fi

# Construct the command
CMD="python odoo_docs_generator.py --addons-path $ADDONS_PATH --output-path $OUTPUT_PATH --api-key $OPENAI_API_KEY --model $MODEL --max-workers $MAX_WORKERS --max-file-tokens $MAX_FILE_TOKENS --batch-size $BATCH_SIZE --rate-limit $RATE_LIMIT"

# Add base_url if specified
if [ ! -z "$BASE_URL" ]; then
    CMD="$CMD --base-url $BASE_URL"
fi

# Add merge-only flag if specified
if [ $MERGE_ONLY -eq 1 ]; then
    CMD="$CMD --merge-only"
fi

# Print the command (without API key)
DISPLAY_CMD=$(echo "$CMD" | sed "s/$OPENAI_API_KEY/*****/")
echo "Running: $DISPLAY_CMD"

# Print proxy settings if set
if [ ! -z "$HTTP_PROXY" ]; then
    echo "Using HTTP proxy: $HTTP_PROXY"
fi
if [ ! -z "$HTTPS_PROXY" ]; then
    echo "Using HTTPS proxy: $HTTPS_PROXY"
fi

# Execute the command
$CMD 