#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SRC_DIR="processed_criteria"
CHUNKS=6
CHUNK_PREFIX="criteria_part"

# Create a working directory for chunks
mkdir -p zip_chunks

# Step 1: List top-level folders and count them
folders=($(find "$SRC_DIR" -mindepth 1 -maxdepth 1 -type d))
total=${#folders[@]}
per_chunk=$(( (total + CHUNKS - 1) / CHUNKS ))  # Round up

# Step 2: Divide folders into 6 chunks and zip each
info() { echo -e "\033[0;32m[INFO]\033[0m $*"; }

info "Total subfolders: $total"
info "Creating $CHUNKS zip chunks, ~${per_chunk} folders each..."

for ((i=0; i<CHUNKS; i++)); do
  chunk_folders=("${folders[@]:i*per_chunk:per_chunk}")
  zipname="zip_chunks/${CHUNK_PREFIX}_${i}.zip"
  info "Creating $zipname with ${#chunk_folders[@]} folders..."

  zip -qr "$zipname" "${chunk_folders[@]}"
done

info "✅ Zipping complete. You can extract all zips into the same target folder."
