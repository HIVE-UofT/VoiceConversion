from datasets import load_from_disk
import torch

# ds_processed = load_from_disk("../CUCO/processed_dataset")
print(ds_processed)
print(f"Number of samples in the dataset: {len(ds_processed)}")
 print("Columns in the dataset:", ds_processed.column_names)
 print("Example sample:", ds_processed[0]['audio_dir_pre'], ds_processed[0]['audio_dir_post'], ds_processed[0]['surgery_type'], ds_processed[0]['file_id'])


