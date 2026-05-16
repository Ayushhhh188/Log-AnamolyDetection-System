import pandas as pd
import re
import os
from datetime import datetime

# Input folder containing all raw .log files
input_folder = "data/raw/HDFS.log"
output_file = "data/processed/parsed_logs.csv"

data = []

# Regex for your actual log format
log_pattern = re.compile(
    r'^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2},\d{3})\s+(\w+)\s+([^:]+):\s+(.*)$'
)

for filename in os.listdir(input_folder):
    if filename.endswith(".log"):
        file_path = os.path.join(input_folder, filename)
        print(f"Processing: {filename}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()

                match = log_pattern.match(line)
                if match:
                    date, time, level, component, message = match.groups()

                    # Convert timestamp
                    timestamp = datetime.strptime(
                        date + " " + time,
                        "%Y-%m-%d %H:%M:%S,%f"
                    )

                    data.append({
                        "timestamp": timestamp,
                        "process_id": None,   # Not available in this log format
                        "log_level": level,
                        "component": component,
                        "content": message
                    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save output
os.makedirs("data/processed", exist_ok=True)
df.to_csv(output_file, index=False)

print(f"Parsing complete! Saved to {output_file}")
print(df.head())
print(f"Total parsed rows: {len(df)}")