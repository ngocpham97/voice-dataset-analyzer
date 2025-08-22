#!/usr/bin/env python3

import os

from datasets import load_dataset

print("🔍 Investigating dataset structure...")

# Use cached dataset
cache_dir = "dataset"
test_url = "https://huggingface.co/datasets/linhtran92/viet_bud500/resolve/main/data/test-00000-of-00002-531c1d81edb57297.parquet"
data_files = {"test": test_url}

dataset = load_dataset("parquet", data_files=data_files, cache_dir=cache_dir)
test_data = dataset["test"]

print(f"Dataset loaded: {len(test_data)} samples")
print("\nDataset features:")
print(test_data.features)

print("\nFirst sample structure:")
sample = test_data[0]
print("Sample keys:", list(sample.keys()))

for key, value in sample.items():
    if isinstance(value, dict):
        print(f"\n{key} (dict):")
        for subkey, subvalue in value.items():
            print(f"  {subkey}: {type(subvalue)} - {str(subvalue)[:100]}...")
    elif hasattr(value, '__len__') and len(str(value)) > 100:
        print(f"{key}: {type(value)} - {str(value)[:100]}...")
    else:
        print(f"{key}: {type(value)} - {value}")
        print(f"{key}: {type(value)} - {value}")
        print(f"{key}: {type(value)} - {value}")
        print(f"{key}: {type(value)} - {value}")
        print(f"{key}: {type(value)} - {value}")
