#!/bin/bash
echo "🔍 MinerU Models Debug Information"
echo "================================="

HF_CACHE="/root/.cache/huggingface"
echo "📁 HuggingFace Cache Directory: $HF_CACHE"
if [ -d "$HF_CACHE" ]; then
    du -sh "$HF_CACHE" 2>/dev/null || echo "Size: Unknown"
    echo "📦 Hub Contents:"
    ls -la "$HF_CACHE/hub/" 2>/dev/null || echo "Hub directory not found"
    
    MODEL_DIR="$HF_CACHE/hub/models--opendatalab--PDF-Extract-Kit-1.0"
    if [ -d "$MODEL_DIR" ]; then
        echo "✅ Model directory exists"
        echo "📋 Model Structure:"
        find "$MODEL_DIR" -type d -maxdepth 3 | sort
        echo "📁 Model files in snapshots:"
        find "$MODEL_DIR/snapshots" -name "*.json" -o -name "*.bin" -o -name "*.safetensors" | head -10
    else
        echo "❌ Model directory not found: $MODEL_DIR"
    fi
else
    echo "❌ Cache directory not found"
fi

CONFIG_FILE="/root/magic-pdf.json"
echo "🔧 MinerU Configuration: $CONFIG_FILE"
if [ -f "$CONFIG_FILE" ]; then
    echo "✅ Config exists"
    cat "$CONFIG_FILE" | head -10
else
    echo "❌ Config not found"
fi