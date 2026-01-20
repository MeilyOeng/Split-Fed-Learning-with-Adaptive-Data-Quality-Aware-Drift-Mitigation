#!/bin/bash

# Script to run a complete training round locally
# This starts: stream service, server, and client

set -e  # Exit on error

echo "=========================================="
echo "Starting Split-Fed Learning Training"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Kill any existing processes on these ports
echo -e "${BLUE}Cleaning up existing processes...${NC}"
lsof -ti:5001 | xargs kill -9 2>/dev/null || true
lsof -ti:5002 | xargs kill -9 2>/dev/null || true
sleep 2

# Set environment variables
export CLIENT_ID="9900002"
export CLUSTER_ID="0"
export DATA_STREAM_API_URL="http://localhost:5001"
export SERVER_PORT="5002"
export CLUSTER_ID_SERVER="0"
export MODEL_INPUT_SIZE="1"
export MODEL_HIDDEN_SIZE="32"
export MODEL_NUM_LAYERS="2"
export MODEL_SPLIT_LAYER="1"
export MODEL_OUTPUT_SIZE="1"
export LEARNING_RATE="0.001"
export BATCH_SIZE="32"
export NUM_EPOCHS="1"
export WINDOW_SIZE="24"

echo -e "${GREEN}Environment variables set${NC}"
echo "  CLIENT_ID: $CLIENT_ID"
echo "  CLUSTER_ID: $CLUSTER_ID"
echo "  SERVER_PORT: $SERVER_PORT"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  NUM_EPOCHS: $NUM_EPOCHS"
echo "  WINDOW_SIZE: $WINDOW_SIZE"

# Start stream service in background
echo -e "\n${BLUE}Starting stream service on port 5001...${NC}"
cd data
python stream_service.py > ../output/stream_service.log 2>&1 &
STREAM_PID=$!
cd ..
sleep 3

# Check if stream service started
if ! kill -0 $STREAM_PID 2>/dev/null; then
    echo "ERROR: Stream service failed to start"
    exit 1
fi
echo -e "${GREEN}Stream service started (PID: $STREAM_PID)${NC}"

# Start server in background
echo -e "\n${BLUE}Starting server on port $SERVER_PORT...${NC}"
cd server
python server.py > ../output/server.log 2>&1 &
SERVER_PID=$!
cd ..
sleep 3

# Check if server started
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    kill $STREAM_PID 2>/dev/null || true
    exit 1
fi
echo -e "${GREEN}Server started (PID: $SERVER_PID)${NC}"

# Wait a bit more for services to be ready
sleep 2

# Run client training
echo -e "\n${BLUE}Starting client training...${NC}"
echo "=========================================="
cd client
python client.py train
CLIENT_EXIT_CODE=$?
cd ..

# Cleanup
echo -e "\n${BLUE}Cleaning up...${NC}"
kill $STREAM_PID 2>/dev/null || true
kill $SERVER_PID 2>/dev/null || true
sleep 1

# Check exit code
if [ $CLIENT_EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}=========================================="
    echo "Training completed successfully!"
    echo "==========================================${NC}"
    echo "Check logs in output/ directory"
else
    echo -e "\n${BLUE}=========================================="
    echo "Training completed with exit code: $CLIENT_EXIT_CODE"
    echo "==========================================${NC}"
    echo "Check logs in output/ directory for details"
fi

exit $CLIENT_EXIT_CODE
