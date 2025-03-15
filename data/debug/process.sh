#!/bin/bash

# Current directory is data/
# First navigate to 3RScan directory
cd 3RScan || { echo "3RScan directory not found!"; exit 1; }
echo "Changed to directory: $(pwd)"

# This script will unzip sequence.zip to a 'sequence' folder in each subfolder
# Skip the 3DSSG_subset subfolder

# Find all directories except 3DSSG_subset
for dir in */; do
  # Skip the 3DSSG_subset directory
  if [ "$dir" != "3DSSG_subset/" ]; then
    echo "Processing directory: $dir"
    
    # Check if sequence folder already exists
    if [ -d "${dir}sequence" ]; then
      echo "Sequence folder already exists in $dir, skipping unzip."
      
      # If sequence.zip exists, delete it
      if [ -f "${dir}sequence.zip" ]; then
        echo "Deleting ${dir}sequence.zip"
        rm "${dir}sequence.zip"
      fi
    # Check if sequence.zip exists in this directory
    elif [ -f "${dir}sequence.zip" ]; then
      # Create sequence directory
      mkdir -p "${dir}sequence"
      
      # Unzip the file into the sequence directory
      echo "Unzipping ${dir}sequence.zip to ${dir}sequence/"
      if unzip -q "${dir}sequence.zip" -d "${dir}sequence/"; then
        # Only delete the zip file if unzip completed successfully
        echo "Unzip completed successfully."
        echo "Deleting ${dir}sequence.zip"
        rm "${dir}sequence.zip"
      else
        echo "Error occurred while unzipping ${dir}sequence.zip. Zip file retained."
      fi
      
      echo "Finished processing $dir"
    else
      echo "No sequence.zip found in $dir, skipping."
    fi
  else
    echo "Skipping 3DSSG_subset directory as requested."
  fi
done

echo "All directories processed."