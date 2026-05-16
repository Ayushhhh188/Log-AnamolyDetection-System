import pandas as pd

# Load parsed logs
input_file = "data/processed/parsed_logs.csv"
output_file = "data/processed/parsed_logs_sample.csv"

# Read only first 100000 rows (adjust later if needed)
df = pd.read_csv(input_file, nrows=100000)

# Save sample
df.to_csv(output_file, index=False)

print(f"Sample created! Saved to {output_file}")
print(df.head())
print(f"Rows in sample: {len(df)}")