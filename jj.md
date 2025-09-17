#!/bin/bash
set -euxo pipefail

# Choose Intel MKL version (oneAPI standalone installer)
MKL_VERSION=2024.0.0
MKL_URL="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/cd7db20d-d7e5-49f0-9e7f-8fb2c8b4cf53/l_onemkl_p_${MKL_VERSION}_offline.sh"

INSTALL_DIR=/opt/intel/oneapi

# Make sure install dir exists
mkdir -p ${INSTALL_DIR}

# Download MKL installer tarball
curl -L -o /tmp/mkl_installer.sh "$MKL_URL"
chmod +x /tmp/mkl_installer.sh

# Install MKL silently (only MKL component)
bash /tmp/mkl_installer.sh -s -a --eula accept --components=intel.oneapi.lin.mkl --install-dir ${INSTALL_DIR}

# Cleanup
rm -f /tmp/mkl_installer.sh

# Set environment so MKL is available to all users
cat << 'EOF' >> /etc/profile.d/mkl.sh
export MKLROOT=/opt/intel/oneapi/mkl/latest
export LD_LIBRARY_PATH=\$MKLROOT/lib/intel64:\$LD_LIBRARY_PATH
export CPATH=\$MKLROOT/include:\$CPATH
export LIBRARY_PATH=\$MKLROOT/lib/intel64:\$LIBRARY_PATH
export MKL_NUM_THREADS=1
EOF
