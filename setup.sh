#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

#=== CONFIGURATION ===#
DATA_URL_1="https://zenodo.org/records/15516900/files/processed_trials.tar.gz?download=1"
RESOURCES_URL="https://zenodo.org/records/15516900/files/resources.tar.gz?download=1"
MODELS_URL="https://zenodo.org/records/15516900/files/models.tar.gz?download=1"
CRITERIA_ZIP_BASE_URL="https://zenodo.org/records/15516900/files"
CHUNK_PREFIX="criteria_part"
CHUNK_COUNT=6

ARCHIVE_1="processed_trials.tar.gz"
RESOURCES_ARCHIVE="resources.tar.gz"
MODELS_ARCHIVE="models.tar.gz"

#=== COLORS ===#
GREEN='\033[0;32m'
NC='\033[0m' # No Color

#=== HELPERS ===#
info() { echo -e "${GREEN}[INFO]${NC} $*"; }
error() { echo -e "[ERROR] $*" >&2; exit 1; }

# Check if file exists and has expected size (within 1% tolerance)
check_file_complete() {
  local file="$1"
  local expected_size="$2"

  if [ ! -f "$file" ]; then
    return 1
  fi

  if [ -z "$expected_size" ]; then
    # No size check, just verify file exists
    return 0
  fi

  local actual_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
  local min_size=$((expected_size * 99 / 100))
  local max_size=$((expected_size * 101 / 100))

  if [ "$actual_size" -ge "$min_size" ] && [ "$actual_size" -le "$max_size" ]; then
    return 0
  else
    info "$file exists but size mismatch (expected ~$expected_size, got $actual_size). Will re-download."
    return 1
  fi
}

#=== MAIN SCRIPT ===#
info "Starting TrialMatchAI setup..."

# 0) Check for available GPUs
info "Checking for available GPUs..."

if command -v nvidia-smi &> /dev/null; then
  if nvidia-smi &> /dev/null; then
    info "NVIDIA GPUs detected:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
  else
    info "nvidia-smi found, but no NVIDIA GPU detected or driver not loaded."
  fi
else
  info "No NVIDIA GPUs detected."
fi

# 1) Install Python dependencies
if ! command -v pip &> /dev/null; then
  error "pip not found. Please install Python and pip first."
fi
info "Installing Python requirements (this will take several minutes)..."
pip install --upgrade pip
info "Installing packages from requirements.txt..."
pip install -r requirements.txt --progress-bar on
info "Python packages installed successfully!"

# 2) Prepare data directory
info "Preparing data directory..."
mkdir -p data
cd data

# Download core archives with size verification
# Expected sizes (in bytes) - update these with actual values or leave empty to skip size check
EXPECTED_SIZE_TRIALS=""      # e.g., 1234567890
EXPECTED_SIZE_RESOURCES=""   # e.g., 1234567890
EXPECTED_SIZE_MODELS=""      # e.g., 1234567890

if check_file_complete "$ARCHIVE_1" "$EXPECTED_SIZE_TRIALS"; then
  info "${ARCHIVE_1} already exists and appears complete. Skipping download."
else
  info "Downloading ${ARCHIVE_1} (this may take several minutes)..."
  wget --continue --progress=bar:force:noscroll "$DATA_URL_1" -O "$ARCHIVE_1" 2>&1 | \
    grep --line-buffered -E "%" || wget --continue --show-progress "$DATA_URL_1" -O "$ARCHIVE_1"
  info "Download of ${ARCHIVE_1} complete!"
fi

if check_file_complete "$RESOURCES_ARCHIVE" "$EXPECTED_SIZE_RESOURCES"; then
  info "${RESOURCES_ARCHIVE} already exists and appears complete. Skipping download."
else
  info "Downloading ${RESOURCES_ARCHIVE} (this may take several minutes)..."
  wget --continue --progress=bar:force:noscroll "$RESOURCES_URL" -O "$RESOURCES_ARCHIVE" 2>&1 | \
    grep --line-buffered -E "%" || wget --continue --show-progress "$RESOURCES_URL" -O "$RESOURCES_ARCHIVE"
  info "Download of ${RESOURCES_ARCHIVE} complete!"
fi

if check_file_complete "$MODELS_ARCHIVE" "$EXPECTED_SIZE_MODELS"; then
  info "${MODELS_ARCHIVE} already exists and appears complete. Skipping download."
else
  info "Downloading ${MODELS_ARCHIVE} (this may take several minutes)..."
  wget --continue --progress=bar:force:noscroll "$MODELS_URL" -O "$MODELS_ARCHIVE" 2>&1 | \
    grep --line-buffered -E "%" || wget --continue --show-progress "$MODELS_URL" -O "$MODELS_ARCHIVE"
  info "Download of ${MODELS_ARCHIVE} complete!"
fi

# Download and extract processed_criteria ZIP chunks
if [ ! -d "processed_criteria" ]; then
  info "Downloading and extracting processed_criteria chunks (${CHUNK_COUNT} files)..."
  mkdir -p processed_criteria

  for i in $(seq 0 $((CHUNK_COUNT - 1))); do
    chunk_zip="${CHUNK_PREFIX}_${i}.zip"
    chunk_url="${CRITERIA_ZIP_BASE_URL}/${chunk_zip}?download=1"

    if check_file_complete "$chunk_zip" ""; then
      info "[$((i+1))/${CHUNK_COUNT}] $chunk_zip already exists. Skipping download."
    else
      info "[$((i+1))/${CHUNK_COUNT}] Downloading $chunk_zip..."
      wget --continue --progress=bar:force:noscroll "$chunk_url" -O "$chunk_zip" 2>&1 | \
        grep --line-buffered -E "%" || wget --continue --show-progress "$chunk_url" -O "$chunk_zip"
    fi

    info "[$((i+1))/${CHUNK_COUNT}] Extracting $chunk_zip..."
    unzip -q -o "$chunk_zip" -d processed_criteria
  done
  info "All criteria chunks downloaded and extracted!"
else
  info "processed_criteria already exists. Skipping download and extraction."
fi

# Extract processed_trials
if [ ! -d "processed_trials" ]; then
  info "Extracting $ARCHIVE_1 (this may take a few minutes)..."
  tar -xzf "$ARCHIVE_1"
  info "Extraction of $ARCHIVE_1 complete!"
else
  info "processed_trials already exists. Skipping extraction of $ARCHIVE_1."
fi

cd ..

# Extract resources
info "Extracting resources into source/Parser..."
mkdir -p source/Parser
tar -xzf data/"$RESOURCES_ARCHIVE" -C source/Parser
info "Resources extracted!"

info "Extracting models into models/..."
mkdir -p models
tar -xzf data/"$MODELS_ARCHIVE" -C models
info "Models extracted!"

info "Cleaning up archives..."
rm -f data/"$ARCHIVE_1" data/"$RESOURCES_ARCHIVE" data/"$MODELS_ARCHIVE"

for i in $(seq 0 $((CHUNK_COUNT - 1))); do
  rm -f data/"${CHUNK_PREFIX}_${i}.zip"
done
info "Cleanup complete!"

# 3) Launch Elasticsearch: Try Docker first, then Apptainer fallback
cd elasticsearch
if command -v docker &> /dev/null && docker info &> /dev/null; then
  info "Docker is available. Setting up Elasticsearch with Docker Compose..."
    docker-compose up -d --build
  cd ..
elif command -v apptainer &> /dev/null; then
  info "Docker not found or not running. Falling back to Apptainer..."
  if [ ! -f "./apptainer-run-es.sh" ]; then
    error "Apptainer script not found at ./elasticsearch/apptainer-run-es.sh"
  fi
  bash ./apptainer-run-es.sh
else
  error "Neither Docker nor Apptainer is available. Cannot continue."
fi
cd ..

# 4) Launch indexers in background
cd utils/Indexer
info "Starting index_criteria.py (trials_eligibility) ..."
nohup python index_criteria.py \
  --config           config.json \
  --processed-folder ../../data/processed_criteria \
  --index-name       trials_eligibility \
  --batch-size       100 \
  --max-workers      100 \
  > criteria.log 2>&1 &

info "Starting index_trials.py (clinical_trials) ..."
nohup python index_trials.py \
  --config           config.json \
  --processed-folder ../../data/processed_trials \
  --index-name       clinical_trials \
  --batch-size       100 \
  > trials.log 2>&1 &

info "Waiting for indexing jobs to complete..."
wait

info "✅ TrialMatchAI setup is complete!"
