#!/bin/bash

# Script to start Docker services and run the UNICEF agent benchmark
# This script assumes you're running it from the unicef-agent directory

set -e  # Exit on any error

echo "🚀 Starting UNICEF Agent Benchmark with Docker Services"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "benchmark/run_benchmark.py" ]; then
    echo -e "${RED}Error: benchmark/run_benchmark.py not found. Please run this script from the unicef-agent directory.${NC}"
    exit 1
fi

# Path to the geospatial directory with docker-compose
GEOSPATIAL_DIR="../unicef-geospatial"

# Check if the geospatial directory exists
if [ ! -d "$GEOSPATIAL_DIR" ]; then
    echo -e "${RED}Error: $GEOSPATIAL_DIR directory not found.${NC}"
    echo "Please ensure the unicef-geospatial directory exists at the same level as unicef-agent."
    exit 1
fi

# Check if docker-compose.yml exists
if [ ! -f "$GEOSPATIAL_DIR/docker-compose.yml" ]; then
    echo -e "${RED}Error: docker-compose.yml not found in $GEOSPATIAL_DIR${NC}"
    exit 1
fi

echo -e "${YELLOW}📦 Starting Docker services...${NC}"
cd "$GEOSPATIAL_DIR"

# Start Docker services
echo "Starting Docker Compose services..."
docker-compose up -d

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 10

# Check if services are running
echo "Checking service status..."
docker-compose ps

# Go back to the agent directory
cd - > /dev/null

echo -e "${YELLOW}🧪 Running benchmark tests...${NC}"

# Run the benchmark using uv
echo "Running benchmark with pytest..."
uv run pytest benchmark/run_benchmark.py -v

echo -e "${GREEN}✅ Benchmark completed!${NC}"
echo ""
echo "📊 Results can be found in:"
echo "  - Numerical results: tests/results/numerical/"
echo "  - Textual results: tests/results/textual/"
echo ""
echo -e "${YELLOW}🛑 To stop Docker services, run:${NC}"
echo "  cd $GEOSPATIAL_DIR && docker-compose down"