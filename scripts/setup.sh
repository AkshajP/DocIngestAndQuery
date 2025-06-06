#!/bin/bash
# Quick Fix for DocRAG Docker Setup
# This script creates necessary directories and fixes volume mount issues

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo -e "${BLUE}🚀 DocRAG Docker Setup Fix${NC}"
echo "=================================="

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOLUMES_DIR="${SCRIPT_DIR}/volumes"

print_info "Working directory: $SCRIPT_DIR"
print_info "Volumes directory: $VOLUMES_DIR"

# Create volumes directory structure
echo
echo "Creating volume directories..."

# Create main volumes directory
mkdir -p "$VOLUMES_DIR"
print_status "Created volumes directory"

# Create all required subdirectories
SUBDIRS=(
    "huggingface_models"
    "postgres"
    "etcd"
    "minio"
    "milvus"
    "pgadmin"
    "paddleocr_models"
)

for dir in "${SUBDIRS[@]}"; do
    mkdir -p "$VOLUMES_DIR/$dir"
    print_status "Created $VOLUMES_DIR/$dir"
done

# Create document_store if it doesn't exist
if [ ! -d "$SCRIPT_DIR/document_store" ]; then
    mkdir -p "$SCRIPT_DIR/document_store"
    print_status "Created document_store directory"
else
    print_info "document_store already exists"
fi

# Set appropriate permissions
echo
echo "Setting permissions..."
chmod -R 755 "$VOLUMES_DIR"
print_status "Set permissions for volumes directory"

# Create a .gitignore for volumes if it doesn't exist
if [ ! -f "$VOLUMES_DIR/.gitignore" ]; then
    cat > "$VOLUMES_DIR/.gitignore" << 'EOF'
# Ignore all volume data but keep the directory structure
*
!.gitignore
!*/
EOF
    print_status "Created volumes/.gitignore"
fi

# Show disk space
echo
echo "Checking available disk space..."
df -h "$SCRIPT_DIR" | tail -1 | awk '{print "Available space: " $4}'

# Show what was created
echo
echo "Directory structure created:"
tree "$VOLUMES_DIR" 2>/dev/null || find "$VOLUMES_DIR" -type d | sort

echo
print_status "Setup completed successfully!"
echo
echo "Next steps:"
echo "1. Run: docker-compose up -d worker"
echo "2. Or pre-download models: ./manage_models.sh download"
echo "3. Check status: ./manage_models.sh status"