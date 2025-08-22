#!/usr/bin/env python3

import json

from datasets import load_dataset

print("🔍 Investigating HuggingFace dataset structure...")

try:
    dataset = load_dataset('linhtran92/viet_bud500', split='train')
    print(f"✓ Dataset loaded successfully: {len(dataset)} samples")

    print("\n📋 Dataset Features:")
    for feature_name, feature_type in dataset.features.items():
        print(f"  {feature_name}: {feature_type}")

    print("\n🔬 First sample analysis:")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")

    for key, value in sample.items():
        if isinstance(value, dict):
            print(f"\n{key} (dict):")
            for subkey, subvalue in value.items():
                if hasattr(subvalue, '__len__') and len(str(subvalue)) > 100:
                    print(f"  {subkey}: {type(subvalue)} - {str(subvalue)[:50]}... (length: {len(subvalue) if hasattr(subvalue, '__len__') else 'N/A'})")
                else:
                    print(f"  {subkey}: {type(subvalue)} - {subvalue}")
        else:
            if hasattr(value, '__len__') and len(str(value)) > 100:
                print(f"{key}: {type(value)} - {str(value)[:50]}... (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
            else:
                print(f"{key}: {type(value)} - {value}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    traceback.print_exc()
    traceback.print_exc()
    traceback.print_exc()
    traceback.print_exc()
