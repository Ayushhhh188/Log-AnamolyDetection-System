import zipfile
import random
import os

zip_path = r"C:\Users\sharm\Downloads\HDFS Dataset.zip"
output_dir = "HDFS.log"
sample_rate = 0.20  # 10%

# Create output folder
os.makedirs(output_dir, exist_ok=True)

with zipfile.ZipFile(zip_path) as z:
    
    # Loop through all files inside zip
    for file in z.namelist():
        
        # Select only HDFS_2k and HDFS_v2 log files
        if ("HDFS_2k" in file or "HDFS_v2" in file) and file.endswith(".log"):
            
            print(f"Processing: {file}")
            
            # Create output file name
            output_file = os.path.join(output_dir, os.path.basename(file))
            
            with z.open(file) as infile, open(output_file, "wb") as outfile:
                for line in infile:
                    if random.random() < sample_rate:
                        outfile.write(line)

print("Sampling complete!")