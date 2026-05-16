"""
Run this first to diagnose exactly why batch_inference fails to import.
    python debug_import.py
"""
import sys, os

BASE_DIR     = os.getcwd()
PIPELINE_DIR = os.path.join(BASE_DIR, "DL", "pipeline")
SIM_DIR      = os.path.join(BASE_DIR, "simulation")

print(f"BASE_DIR     : {BASE_DIR}")
print(f"PIPELINE_DIR : {PIPELINE_DIR}")
print(f"SIM_DIR      : {SIM_DIR}")
print()

# Check paths exist
for label, path in [("PIPELINE_DIR", PIPELINE_DIR), ("SIM_DIR", SIM_DIR)]:
    exists = os.path.isdir(path)
    print(f"  {'✓' if exists else '✗'} {label} exists: {exists}  →  {path}")

print()

# List files in pipeline dir
if os.path.isdir(PIPELINE_DIR):
    files = os.listdir(PIPELINE_DIR)
    print(f"Files in DL/pipeline/:")
    for f in files:
        print(f"    {f}")
else:
    print("DL/pipeline/ directory NOT FOUND")

print()

# Try adding to path and importing
for p in (PIPELINE_DIR, SIM_DIR, BASE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

print("Attempting: from batch_inference import predict_batch ...")
try:
    from batch_inference import predict_batch
    print("  ✓ batch_inference imported successfully")
    print(f"  ✓ predict_batch = {predict_batch}")
except Exception as e:
    print(f"  ✗ FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print()
print("Attempting: from log_generator import LogSimulator ...")
try:
    from log_generator import LogSimulator
    print("  ✓ LogSimulator imported successfully")
except Exception as e:
    print(f"  ✗ FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()