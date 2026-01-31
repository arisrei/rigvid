#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_VER="3.10"
PYTORCH_VER="2.4.1"

# Parse command line arguments
CONTINUE_VENV=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --continue)
            CONTINUE_VENV=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --continue    Continue using existing .venv instead of creating a new one"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Install system dependencies (eigen3 for FoundationPose)
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y libeigen3-dev

# Initialize uv project if not already done
if [ ! -f "pyproject.toml" ]; then
    echo "Initializing uv project..."
    uv init --python "${PYTHON_VER}" --no-workspace
fi

# Create or reuse virtual environment
if [ "$CONTINUE_VENV" = true ] && [ -d ".venv" ]; then
    echo "Continuing with existing virtual environment..."
else
    echo "Creating virtual environment with Python ${PYTHON_VER}..."
    uv venv .venv --python "${PYTHON_VER}"
fi

# Activate the virtual environment for this script
source .venv/bin/activate

# Clone repositories
echo "Cloning RollingDepth and FoundationPose repositories..."
git clone https://github.com/prs-eth/RollingDepth.git || true
git clone https://github.com/NVlabs/FoundationPose.git || true

# Install build dependencies first
echo "Installing build dependencies..."
uv pip install setuptools wheel pip

# Install numpy and scipy with ABI compatibility
echo "Installing numpy and scipy..."
uv pip install numpy==1.26.4 scipy==1.12.0 PyJWT

# Install GPU PyTorch (CUDA 12.1 for compatibility with system CUDA 12.x)
echo "Installing PyTorch, torchvision, and torchaudio..."
uv pip install \
  "torch==${PYTORCH_VER}+cu121" \
  "torchvision==0.19.1+cu121" \
  "torchaudio==${PYTORCH_VER}+cu121" \
  --extra-index-url https://download.pytorch.org/whl/cu121

# Create constraints file
cat > constraints.txt <<EOF
torch==${PYTORCH_VER}+cu121
torchvision==0.19.1+cu121
torchaudio==${PYTORCH_VER}+cu121
numpy==1.26.4
scipy==1.12.0
EOF

# Install RollingDepth dependencies and diffusers development version
echo "Installing RollingDepth dependencies..."
pushd RollingDepth >/dev/null
uv pip install -r requirements.txt -c ../constraints.txt
bash script/install_diffusers_dev.sh
popd >/dev/null

# Set PYTHONPATH for RollingDepth
export PYTHONPATH="$(pwd)/RollingDepth:${PYTHONPATH:-}"

# Install FoundationPose dependencies
echo "Installing FoundationPose dependencies..."

pushd FoundationPose >/dev/null

# Adjust FoundationPose requirements to avoid conflicts
sed -i '/^torch==.*+cu118/d; /^torchvision==.*+cu118/d; /^torchaudio==.*+cu118/d' requirements.txt
uv pip install -r requirements.txt -c ../constraints.txt

# Additional FoundationPose dependencies
# Use venv's pip directly for packages that require PyTorch during build
# (uv pip's --no-build-isolation doesn't work the same way)
# Set TORCH_CUDA_ARCH_LIST to bypass strict CUDA version check during build
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
../.venv/bin/pip install --no-build-isolation --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
uv pip install fvcore iopath ninja
# Try pre-built wheel first, fall back to source build if not available
# Check https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/ for available wheels
uv pip install pytorch3d --index-url https://dl.fbaipublicfiles.com/pytorch3d/whl/cu121/torch2.4.0/pyt240 --no-deps 2>/dev/null || \
  ../.venv/bin/pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git

# Set CMAKE prefix path for pybind11
PYTHON_SITE_PACKAGES=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || python -c "import site; print(site.getsitepackages()[0])")
export CMAKE_PREFIX_PATH="${PYTHON_SITE_PACKAGES}/pybind11/share/cmake/pybind11:${CMAKE_PREFIX_PATH:-}"

# Configure setup.py for FoundationPose CUDA extensions
SETUP_PY="bundlesdf/mycuda/setup.py"
cat > "$SETUP_PY" <<'EOF'
from setuptools import setup
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

code_dir = os.path.dirname(os.path.realpath(__file__))

cxx_flags = ['-O3', '-std=c++17']
nvcc_flags = [
    '-O3',
    '--expt-relaxed-constexpr',
    '-std=c++17',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
]

setup(
    name='common',
    ext_modules=[
        CUDAExtension('common', [
            'bindings.cpp',
            'common.cu',
        ], extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags}),
        CUDAExtension('gridencoder', [
            f"{code_dir}/torch_ngp_grid_encoder/gridencoder.cu",
            f"{code_dir}/torch_ngp_grid_encoder/bindings.cpp",
        ], extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags}),
    ],
    include_dirs=[
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
EOF

# Build FoundationPose extensions
bash build_all_conda.sh
popd >/dev/null

# Completion message
echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Run the projects with:"
echo "  RollingDepth: python run_video.py"
echo "  FoundationPose: cd FoundationPose && python run_demo.py"
