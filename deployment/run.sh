#!/bin/bash
set -e

uvicorn api:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

# Ensure the background API process is cleaned up if this script exits for any reason
trap "kill $UVICORN_PID" EXIT

echo "Waiting for the API to become ready..."
for i in $(seq 1 30); do
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health | grep -q "200"; then
        echo "API is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "API did not become ready in time." >&2
        exit 1
    fi
    sleep 1
done

streamlit run app.py --server.port 7860 --server.address 0.0.0.0