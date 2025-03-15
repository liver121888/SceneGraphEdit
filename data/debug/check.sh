# !/bin/bash

# Navigate to 3RScan directory
cd 3RScan || { echo "3RScan directory not found!"; exit 1; }
echo "Changed to directory: $(pwd)"

# Create a log file for corrupted hashes
log_file="../corrupted_hashes.log"
echo "Data integrity check started at $(date)" > "$log_file"

# Find all directories except 3DSSG_subset
for dir in */; do
  # Skip the 3DSSG_subset directory
  if [ "$dir" != "3DSSG_subset/" ]; then
    echo "Checking directory: $dir"
    
    # Check if sequence folder exists
    if [ -d "${dir}sequence" ]; then
      # Check if _info.txt exists
      if [ -f "${dir}sequence/_info.txt" ]; then
        # Extract m_frames.size value
        frames_size=$(grep "m_frames.size" "${dir}sequence/_info.txt" | awk -F'=' '{print $2}' | tr -d ' ')
        
        if [ -z "$frames_size" ]; then
          echo "ERROR: Could not find m_frames.size in ${dir}sequence/_info.txt"
          echo "${dir}: Missing m_frames.size information" >> "$log_file"
          # echo "Deleting corrupted hash folder: $dir"
          # rm -rf "$dir"
          continue
        fi
        
        echo "Found m_frames.size = $frames_size in ${dir}"
        
        # Calculate expected number of files (3 files per frame)
        expected_files=$((frames_size * 3))
        
        # Count actual files (color, depth, pose)
        color_files=$(find "${dir}sequence" -name "frame-*.color.jpg" | wc -l)
        depth_files=$(find "${dir}sequence" -name "frame-*.depth.pgm" | wc -l)
        pose_files=$(find "${dir}sequence" -name "frame-*.pose.txt" | wc -l)
        total_files=$((color_files + depth_files + pose_files))
        
        echo "Expected files: $expected_files, Found: $total_files (${color_files} color, ${depth_files} depth, ${pose_files} pose)"
        
        # Check for missing frames
        corrupted=false
        for ((i=0; i<frames_size; i++)); do
          # Format frame number with leading zeros (6 digits)
          frame_num=$(printf "%06d" $i)
          
          # Check if all three files exist for this frame
          if [ ! -f "${dir}sequence/frame-${frame_num}.color.jpg" ] || 
             [ ! -f "${dir}sequence/frame-${frame_num}.depth.pgm" ] || 
             [ ! -f "${dir}sequence/frame-${frame_num}.pose.txt" ]; then
            echo "ERROR: Files missing in ${dir}"
            echo "${dir}: Corrupted sequence data" >> "$log_file"
            corrupted=true
            break
          fi
        done
        
        # Check if total file count matches expected
        if [ $total_files -ne $expected_files ] || [ "$corrupted" = true ]; then
          echo "ERROR: Corrupted data in ${dir}"
          echo "${dir}: Expected $expected_files files, found $total_files" >> "$log_file"
          # echo "Deleting corrupted hash folder: $dir"
          # rm -rf "$dir"
        else
          echo "SUCCESS: All files present for ${dir}"
        fi
      else
        echo "ERROR: _info.txt not found in ${dir}sequence/"
        echo "${dir}: Missing _info.txt" >> "$log_file"
        # echo "Deleting corrupted hash folder: $dir"
        # rm -rf "$dir"
      fi
    else
      echo "ERROR: sequence folder not found in ${dir}"
      echo "${dir}: Missing sequence folder" >> "$log_file"
      # echo "Deleting corrupted hash folder: $dir"
      # rm -rf "$dir"
    fi
    
    echo "-----------------------------------"
  else
    echo "Skipping 3DSSG_subset directory as requested."
  fi
done

# Count the number of corrupted hashes
corrupted_count=$(grep -c ":" "$log_file")

echo "Verification complete."
echo "Total corrupted hash folders: $corrupted_count"
echo "Check $log_file for the list of corrupted hash folders."


# # Navigate to 3RScan directory
# # Remove corrupted hashes
# cd 3RScan || { echo "3RScan directory not found!"; exit 1; }
# echo "Changed to directory: $(pwd)"

# # List of corrupted hash folders to remove
# corrupted_folders=(
#   "7ab2a9cb-ebc6-2056-8ba2-7835e43d47d3"
#   "7f30f368-42f9-27ed-852b-e6cfc067acea"
#   "80b8588d-4a8d-222f-84c5-8c51b9af1c2f"
#   "80b85891-4a8d-222f-85d0-2b8bf16d79a8"
#   "80b85893-4a8d-222f-8519-a9d06c205653"
#   "82def5da-25ba-20f9-8810-71be905b83db"
#   "87e6cf6b-9d1a-289f-866c-b90904d9487d"
#   "87e6cf6d-9d1a-289f-879a-543d3fa7ba74"
#   "87e6cf6f-9d1a-289f-8693-db8b73a4c4f4"
#   "87e6cf71-9d1a-289f-8510-bddeda7aaad8"
#   "87e6cf7b-9d1a-289f-8692-57e5757dac99"
# )

# # Log file to record the removal operations
# removal_log="../removal_operations.log"
# echo "Removal operations started at $(date)" > "$removal_log"

# # Counter for successfully removed folders
# removed_count=0

# # Remove each corrupted folder
# for folder in "${corrupted_folders[@]}"; do
#   if [ -d "$folder" ]; then
#     echo "Removing folder: $folder"
#     # Remove the folder
#     rm -rf "$folder"
    
#     # Check if removal was successful
#     if [ ! -d "$folder" ]; then
#       echo "Successfully removed $folder" >> "$removal_log"
#       ((removed_count++))
#     else
#       echo "Failed to remove $folder" >> "$removal_log"
#     fi
#   else
#     echo "Folder not found: $folder (already removed or never existed)" >> "$removal_log"
#   fi
# done

# echo "Removal operations complete."
# echo "Successfully removed $removed_count folders."
# echo "Check $removal_log for details."