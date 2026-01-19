import os
from pathlib import Path
import pandas as pd
from datasets import Dataset, Audio
import librosa
import numpy as np

data_root = Path("../CUCO/data_final/Audios")

data_list = []

surgeries = ["Fess", "Sept", "Tonsill"]

for surgery in surgeries:
    surgery_path = data_root / surgery / "Speech"
    for folder_num in ["1"]:
        folder_path = surgery_path / folder_num
        
        if not folder_path.exists():
            continue
            
        for audio_file in folder_path.glob("*.wav"):   
            data_list.append({
                "audio_pre": str(audio_file),
                "audio_post": str(audio_file).replace("_ses1_", "_ses2_").replace("/1/", "/2/"),
                "audio_dir_pre": str(audio_file),
                "audio_dir_post": str(audio_file).replace("_ses1_", "_ses2_").replace("/1/", "/2/"),
                "surgery_type": surgery.lower(),
                "file_id": audio_file.stem 
            })

ds = Dataset.from_list(data_list)

ds = ds.cast_column("audio_pre", Audio(sampling_rate=16000))
ds = ds.cast_column("audio_post", Audio(sampling_rate=16000))

def compute_dual_mels(batch):
    
    pre_arrays = [x["array"] for x in batch["audio_pre"]]
    post_arrays = [x["array"] for x in batch["audio_post"]]
    sr = batch["audio_pre"][0]["sampling_rate"] 
    
    mels_pre = []
    mels_post = []
    
    for y_pre, y_post in zip(pre_arrays, post_arrays):
        S_pre = librosa.feature.melspectrogram(y=y_pre, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
        mels_pre.append(librosa.power_to_db(S_pre, ref=np.max))
        
        S_post = librosa.feature.melspectrogram(y=y_post, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
        mels_post.append(librosa.power_to_db(S_post, ref=np.max))
        
    batch["mel_pre"] = mels_pre
    batch["mel_post"] = mels_post
    return batch

ds_processed = ds.map(compute_dual_mels, batched=True, batch_size=8)

ds_processed.save_to_disk("../CUCO/processed_dataset")