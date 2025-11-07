#!/bin/bash
URL="http://127.0.0.1:8000"
curl -s $URL/health | jq .
curl -s -X POST "$URL/recommend" -H "Content-Type: application/json" \
 -d '{"job_title":"Java developer who collaborates with teams","top_k":5}' | jq .
