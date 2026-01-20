#!/bin/bash
# Start Chroma API Server with configurable dp-size

# Default values
HOST="0.0.0.0"
PORT=8000
CHROMA_MODEL_PATH=""
DP_SIZE=1
WORKERS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --chroma-model-path)
            CHROMA_MODEL_PATH="$2"
            shift 2
            ;;
        --dp-size)
            DP_SIZE="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST                 Host to bind (default: 0.0.0.0)"
            echo "  --port PORT                 Port to bind (default: 8000)"
            echo "  --chroma-model-path PATH    Path to Chroma model"
            echo "  --dp-size SIZE              Data parallel size (default: 1)"
            echo "  --workers NUM               Number of workers (default: 1)"
            echo "  -h, --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Chroma API Server"
echo "=========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "DP Size: $DP_SIZE"
echo "Workers: $WORKERS"
echo "Chroma Model: $CHROMA_MODEL_PATH"
echo "=========================================="

# Validate required parameters
if [ -z "$CHROMA_MODEL_PATH" ]; then
    echo "Error: --chroma-model-path is required"
    echo "Use --help for usage information"
    exit 1
fi

# Export environment variables
export CHROMA_MODEL_PATH="$CHROMA_MODEL_PATH"
export DP_SIZE="$DP_SIZE"

# Check if we need distributed launch
if [ $DP_SIZE -gt 1 ]; then
    echo "Launching with distributed training (dp_size=$DP_SIZE)..."
    
    # Use torchrun for distributed launch
    torchrun \
        --nproc_per_node=$DP_SIZE \
        --master_addr=127.0.0.1 \
        --master_port=29500 \
        api_server.py \
        --host "$HOST" \
        --port "$PORT" \
        --chroma-model-path "$CHROMA_MODEL_PATH" \
        --dp-size "$DP_SIZE" \
        --workers "$WORKERS"
else
    echo "Launching in single-node mode..."

    # Single node launch
    python api_server.py \
        --host "$HOST" \
        --port "$PORT" \
        --chroma-model-path "$CHROMA_MODEL_PATH" \
        --dp-size "$DP_SIZE" \
        --workers "$WORKERS"
fi
