import sys
import os
sys.path.insert(0, os.path.abspath("."))
from src.step60_sota_pipeline import run_step60_evaluation

if __name__ == "__main__":
    print("Starting Step 60 SOTA Model Training & Holdout Benchmark...")
    results = run_step60_evaluation()
    print("Training finished successfully! Results:")
    print(results)
