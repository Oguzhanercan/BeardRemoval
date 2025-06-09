#!/bin/bash

ZIP_URL=$1         # URL of the .zip file
DEST_DIR=$2        # Target directory to extract to

if [ -z "$ZIP_URL" ] || [ -z "$DEST_DIR" ]; then
  echo "Usage: $0 <zip_url> <destination_directory>"
  exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Temporary zip file path
TEMP_ZIP=$(mktemp)

# Download the zip
echo "Downloading $ZIP_URL..."
wget -O "$TEMP_ZIP" "$ZIP_URL" || { echo "Download failed"; exit 1; }

# Extract it
echo "Extracting to $DEST_DIR..."
unzip -q "$TEMP_ZIP" -d "$DEST_DIR" || { echo "Extraction failed"; exit 1; }

# Clean up
rm "$TEMP_ZIP"

echo "Done. Extracted contents to $DEST_DIR"