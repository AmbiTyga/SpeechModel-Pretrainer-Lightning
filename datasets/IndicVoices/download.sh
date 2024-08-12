#!/bin/bash

# Define the text file containing URLs
url_file="indicVoices.train.txt"
error_file="failed_downloads.txt"

# Check if the input file exists
if [[ ! -f "$url_file" ]]; then
  echo "File $url_file not found!"
  exit 1
fi

# Clear or create the error file
> "$error_file"

# Function to download files using curl and handle errors
download() {
  local url="$1"
  local output_file="$(basename "$url")"

  # Remove the token part from the filename
  output_file="${output_file%%\?*}"

  # Download the file and check for errors
  if curl -f -s -o "$output_file" "$url"; then
    echo "Downloaded: $url"
  else
    echo "Failed to download: $url" | tee -a "$error_file"
  fi
}

# Read each URL from the file and download it
while IFS= read -r url
do
  if [[ -n "$url" ]]; then
    download "$url"
  fi
done < "$url_file"

echo "Download completed. Check $error_file for failed downloads."
