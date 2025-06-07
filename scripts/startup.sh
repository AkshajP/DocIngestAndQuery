#!/bin/bash
set -e

echo "🚀 Starting DocRAG Backend Application"

# Function to check if models are already downloaded
check_models_exist() {
    local model_dir="/root/.cache/huggingface/hub/models--opendatalab--PDF-Extract-Kit-1.0"
    local config_file="/root/magic-pdf.json"
    
    if [ -f "$config_file" ] && [ -d "$model_dir" ]; then
        echo "✅ Models and config already exist, checking completeness..."
        
        # Check if snapshots directory exists and has content
        local snapshots_dir="$model_dir/snapshots"
        if [ ! -d "$snapshots_dir" ]; then
            echo "❌ Snapshots directory missing: $snapshots_dir"
            return 1
        fi
        
        # Find the actual snapshot directory (there should be one with a hash name)
        local snapshot_dirs=($(find "$snapshots_dir" -mindepth 1 -maxdepth 1 -type d))
        if [ ${#snapshot_dirs[@]} -eq 0 ]; then
            echo "❌ No snapshot directories found in $snapshots_dir"
            return 1
        fi
        
        local snapshot_dir="${snapshot_dirs[0]}"
        echo "🔍 Checking snapshot: $(basename "$snapshot_dir")"
        
        # Check for models directory
        local models_dir="$snapshot_dir/models"
        if [ ! -d "$models_dir" ]; then
            echo "❌ Models directory missing: $models_dir"
            return 1
        fi
        
        # Check for critical model subdirectories (more flexible approach)
        local required_models=("Layout" "MFD" "MFR")
        local missing_count=0
        
        for model_name in "${required_models[@]}"; do
            if [ -d "$models_dir/$model_name" ]; then
                echo "✅ $model_name model directory found"
                # Check if it has some content (not empty)
                if [ "$(ls -A "$models_dir/$model_name" 2>/dev/null | wc -l)" -gt 0 ]; then
                    echo "✅ $model_name has content"
                else
                    echo "⚠️  $model_name directory is empty"
                    missing_count=$((missing_count + 1))
                fi
            else
                echo "❌ $model_name model directory missing"
                missing_count=$((missing_count + 1))
            fi
        done
        
        # Check total size as additional verification
        local total_size
        if command -v du >/dev/null 2>&1; then
            total_size=$(du -sm "$model_dir" 2>/dev/null | cut -f1)
            echo "📊 Total model size: ${total_size}MB"
            
            # PDF-Extract-Kit should be at least 50MB when fully downloaded
            if [ "$total_size" -lt 50 ]; then
                echo "⚠️  Model size seems too small (${total_size}MB < 50MB expected)"
                missing_count=$((missing_count + 1))
            fi
        fi
        
        if [ $missing_count -eq 0 ]; then
            echo "✅ All model components are complete, skipping download"
            return 0
        else
            echo "⚠️  Some model components are missing or incomplete ($missing_count issues)"
            echo "📂 Model structure:"
            find "$models_dir" -type d -maxdepth 2 2>/dev/null | head -10
            return 1
        fi
    else
        if [ ! -f "$config_file" ]; then
            echo "📥 MinerU config not found: $config_file"
        fi
        if [ ! -d "$model_dir" ]; then
            echo "📥 Model directory not found: $model_dir"
        fi
        return 1
    fi
}

# Function to download models with progress
download_models() {
    echo "📥 Starting MinerU models download..."
    echo "📁 Cache directory: /root/.cache/huggingface"
    echo "💾 Available space:"
    df -h /root/.cache/huggingface | tail -1
    
    # Download with retry logic
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        echo "🔄 Download attempt $((retry_count + 1))/$max_retries"
        
        if python download_models_hf.py; then
            echo "✅ MinerU models download completed successfully!"
            
            # Verify download was successful
            if check_models_exist; then
                echo "✅ Model download verification passed"
                return 0
            else
                echo "❌ Model download verification failed"
                retry_count=$((retry_count + 1))
            fi
        else
            echo "❌ Model download failed (attempt $((retry_count + 1))/$max_retries)"
            retry_count=$((retry_count + 1))
            
            if [ $retry_count -lt $max_retries ]; then
                echo "⏳ Waiting 30 seconds before retry..."
                sleep 30
            fi
        fi
    done
    
    echo "❌ Failed to download models after $max_retries attempts"
    return 1
}

# Function to update config.json with environment variables
update_app_config() {
    if [ -f "config.json" ] && [ ! -z "$DOCRAG_VECTOR_DB_HOST" ]; then
        echo "🔧 Updating application config with environment variables..."
        python3 -c "
import json
import os
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    config['vector_db']['host'] = os.getenv('DOCRAG_VECTOR_DB_HOST', 'localhost')
    config['vector_db']['port'] = os.getenv('DOCRAG_VECTOR_DB_PORT', '19530')
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f'✅ Updated vector_db: {config[\"vector_db\"][\"host\"]}:{config[\"vector_db\"][\"port\"]}')
except Exception as e:
    print(f'⚠️  Config update failed: {e}')
"
    fi
}

# Function to verify MinerU setup
verify_mineru() {
    echo "🔍 Verifying MinerU setup..."
    
    # First check if config file exists
    if [ ! -f "/root/magic-pdf.json" ]; then
        echo "❌ MinerU config file missing: /root/magic-pdf.json"
        return 1
    fi
    
    echo "✅ MinerU config file found"
    
    # Test MinerU imports and functionality
    python3 -c "
import sys
import os

try:
    print('🧪 Testing MinerU imports...')
    
    from magic_pdf.libs.config_reader import read_config
    config = read_config()
    print('✅ MinerU config read successfully')
    
    # Check if model directories exist
    if 'model-dir' in config:
        model_dir = config['model-dir']
        print(f'📂 Model directory from config: {model_dir}')
        if os.path.exists(model_dir):
            print('✅ Model directory exists')
            
            # Check for specific model subdirectories
            required_models = ['Layout', 'MFD', 'MFR']
            missing_models = []
            for model in required_models:
                model_path = os.path.join(model_dir, model)
                if os.path.exists(model_path):
                    file_count = len([f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))])
                    print(f'✅ {model} model found ({file_count} files)')
                else:
                    missing_models.append(model)
                    print(f'❌ {model} model missing')
            
            if missing_models:
                print(f'❌ Missing models: {missing_models}')
                sys.exit(1)
        else:
            print(f'❌ Model directory does not exist: {model_dir}')
            sys.exit(1)
    else:
        print('⚠️  No model-dir found in config')
    
    # Test core imports
    from magic_pdf.data.dataset import PymuDocDataset
    print('✅ PymuDocDataset import successful')
    
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    print('✅ doc_analyze import successful')
    
    print('✅ All MinerU components verified successfully')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ MinerU verification failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" || {
        echo "❌ MinerU verification failed"
        echo "🔍 Running debug information..."
        /app/debug_models.sh
        return 1
    }
    
    echo "✅ MinerU is ready and functional"
    return 0
}

# Main startup sequence
echo "📋 Running startup checks..."
echo "📁 Cache mount info:"
mount | grep huggingface || echo "No specific huggingface mount found"

# Update application configuration
update_app_config

# Check if models exist, download if needed
if ! check_models_exist; then
    if ! download_models; then
        echo "❌ Critical: Failed to download required models"
        exit 1
    fi
else
    echo "🎯 Using existing models from cache"
fi

# Verify MinerU functionality
if ! verify_mineru; then
    echo "❌ Critical: MinerU verification failed"
    exit 1
fi

echo "🔧 Configuring MinerU settings..."
python3 -c "
from download_models_hf import configure_magic_pdf_settings
configure_magic_pdf_settings()
"
echo "set the following environment variables to change any configuration for magic-pdf:
    - MINERU_DEVICE_MODE
    - MINERU_LAYOUT_MODEL
    - MINERU_TABLE_MODEL
    - MINERU_TABLE_ENABLE
    - MINERU_FORMULA_ENABLE
    - MINERU_LLM_ENABLE
    - OLLAMA_MODEL \n"

echo "🌟 All checks passed! Starting application..."

# Start the application based on the command
if [ "$1" = "worker" ]; then
    echo "🔄 Starting Celery worker..."
    exec python worker.py
else
    echo "🌐 Starting FastAPI/Uvicorn server..."
    exec uvicorn main:app --host 0.0.0.0 --port 9000 --reload
fi