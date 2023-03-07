"""snip download"""

import requests
import os
import os.path
import fire
import numpy as np
import pandas as pd
# from huggingface_hub import hf_hub_url

def snip_download(outfolder="data/downloaded", start=0, end=2313, dl_dedup_set=True):
    """Download and deduplicate a dataset.

    Parameters
    ----------
    outfolder : str, optional
        Where to put the downloaded metadata

    start : int, optional
        Start index of the metadata

    end : int, optional
        End index of the metadata

    dl_dedup_set : bool, optional
        Indicate whether you'll download the dedup set again (2GB)
    """
    metadata_dir = os.path.join(outfolder, "metadata")
    dedup_set_path = os.path.join(outfolder, "is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy")
    os.makedirs(metadata_dir, exist_ok=True)

    if dl_dedup_set:
        print('downloading dedup set...')
        url = f'https://huggingface.co/datasets/fraisdufour/snip-dedup/resolve/main/is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy'
        response = requests.get(url)
        # np.save('is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy',response.content)
        # hf_hub_url(repo_id="fraisdufour/snip-dedup", filename="is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy")
        open(dedup_set_path, "wb").write(response.content)
        
    is_dup_all = np.load(dedup_set_path).ravel()
    abs_ind = 0
    for n in range(start,end):
        print(f'downloading metadata file {n}/{end}')
        url = f'https://huggingface.co/datasets/laion/laion2b-en-vit-h-14-embeddings/resolve/main/metadata/metadata_{n:04d}.parquet'
        response = requests.get(url)
        parquet_path = os.path.join(metadata_dir, f'metadata_{n:04d}.parquet')
        open(parquet_path, "wb").write(response.content)
        
        # perform the deduplication
        md = pd.read_parquet(parquet_path)
        non_dup_chunk = is_dup_all[abs_ind:abs_ind+len(md.index)]
        
        # take only non-dupped (uniques)
        non_dup_chunk = np.logical_not(non_dup_chunk)
        
        # make sure there is at least one unique
        non_dup_chunk[0] = True
        
        md = md[non_dup_chunk]
        
        # overwrite metadata
        md.to_parquet(parquet_path)
        abs_ind+=len(md.index)
    
    
if __name__ == "__main__":
    fire.Fire(snip_download)
