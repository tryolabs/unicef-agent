#!/bin/bash

# Benchmark runner script for testing AI model performance
#
# Usage examples:
#   ./run_benchmark.sh                    # Run with default 10 workers
#   ./run_benchmark.sh -n 5               # Run with 5 parallel workers
#   ./run_benchmark.sh --workers 20       # Run with 20 parallel workers

# Default number of parallel workers
WORKERS=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--workers)
            WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            echo ""
            echo "Usage examples:"
            echo "  ./run_benchmark.sh                    # Run with default 10 workers"
            echo "  ./run_benchmark.sh -n 5               # Run with 5 parallel workers"
            echo "  ./run_benchmark.sh --workers 20       # Run with 20 parallel workers"
            exit 1
            ;;
    esac
done

# Start the GEE MCP server in the background
echo "Starting GEE MCP server..."
cd ../unicef-gee-mcp
uv run python ee_mcp/server.py &
GEE_SERVER_PID=$!

# Start the RAG MCP server in the background
echo "Starting RAG MCP server..."
cd ../unicef-rag-mcp
uv run python rag/server.py &
RAG_SERVER_PID=$!

# Start the Datawarehouse MCP server in the background
echo "Starting Datawarehouse MCP server..."
cd ../unicef-datawarehouse-mcp
uv run python datawarehouse_mcp/server.py &
DATAWAREHOUSE_SERVER_PID=$!

cd ../unicef-agent

# Wait a moment for the servers to start up
sleep 5

# Function to cleanup servers on exit
cleanup() {
    echo "Stopping MCP servers..."
    kill $GEE_SERVER_PID 2>/dev/null
    kill $RAG_SERVER_PID 2>/dev/null
    kill $DATAWAREHOUSE_SERVER_PID 2>/dev/null
}
trap cleanup EXIT

# Remove existing session ID to create a fresh one for this benchmark run
rm benchmark/results/.session_id

# Add current directory to PYTHONPATH to ensure tests module can be found
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the benchmark tests with the specified number of parallel workers
uv run pytest benchmark/run_benchmark.py -n $WORKERS