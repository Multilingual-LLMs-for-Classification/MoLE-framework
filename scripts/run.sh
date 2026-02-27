#!/bin/bash
# Run the Classification Service locally (without Docker)

set -e

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Set Python path to include project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Default settings
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"
export DEBUG="${DEBUG:-false}"

# JWT settings (change in production!)
export JWT_SECRET_KEY="${JWT_SECRET_KEY:-dev-secret-key-change-me}"
export JWT_ACCESS_TOKEN_EXPIRE_MINUTES="${JWT_ACCESS_TOKEN_EXPIRE_MINUTES:-30}"

echo "=============================================="
echo "Starting MOE Classification Service"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo "API Host: ${API_HOST}:${API_PORT}"
echo "=============================================="

# Run the service
if [ "$DEBUG" = "true" ]; then
    uvicorn app.main:app --host "${API_HOST}" --port "${API_PORT}" --reload
else
    uvicorn app.main:app --host "${API_HOST}" --port "${API_PORT}"
fi
