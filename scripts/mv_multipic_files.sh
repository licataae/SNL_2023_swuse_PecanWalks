#!/bin/sh

# Define the list of strings
search_strings=("PICTURE_116" "PICTURE_127" "PICTURE_158" "PICTURE_177" "PICTURE_234" "PICTURE_240" "PICTURE_265" "PICTURE_265" "PICTURE_269" "PICTURE_279" "PICTURE_287" "PICTURE_294" "PICTURE_311" "PICTURE_322" "PICTURE_324" "PICTURE_346" "PICTURE_359" "PICTURE_393" "PICTURE_394" "PICTURE_425" "PICTURE_452" "PICTURE_476" "PICTURE_486" "PICTURE_499" "PICTURE_500" "PICTURE_504" "PICTURE_516" "PICTURE_551" "PICTURE_563" "PICTURE_589" "PICTURE_590" "PICTURE_611" "PICTURE_618" "PICTURE_62" "PICTURE_632" "PICTURE_67" "PICTURE_670" "PICTURE_689" "PICTURE_743" "PICTURE_747" "PICTURE_78" "PICTURE_80")

# Define the source and destination directories
source_dir="/Users/licata/Documents/PROJECTS/WPGroundingLabels/behavioral_paradigms/stimulus_dev/pic_databases/MultiPic/IMAGES/colored_JPG"
destination_dir="/Users/licata/Documents/PROJECTS/WPGroundingLabels/behavioral_paradigms/stimulus_dev/stimuli/stimulus_images_B&W"

# Create the destination directory if it doesn't exist
mkdir -p "$destination_dir"

# Loop through the search strings
for string in "${search_strings[@]}"
do
    # Use find to locate matching .png files and copy them to the destination directory
    find "$source_dir" -type f -name "*$string*.jpg" -exec cp {} "$destination_dir" \;
done
