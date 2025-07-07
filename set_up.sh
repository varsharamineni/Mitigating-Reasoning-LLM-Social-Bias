#!/bin/bash
echo "🔧 Starting setup_env.sh..."

# Load compatible Python module (only if needed by your cluster)
module purge
module load profile/base
module load python/3.10.8--gcc--8.5.0

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✅ uv already installed."
fi

# Clone the repository if needed (optional)
# git clone <repository-url>
# cd path/to/project

# Create and sync the virtual environment with uv
echo "📦 Creating and syncing virtual environment using uv..."
uv venv .venv
uv sync

# Activate the environment
source .venv/bin/activate

echo "✅ Environment setup complete."