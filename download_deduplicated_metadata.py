import requests
import os
import fire
import numpy as np
import pandas as pd
# from huggingface_hub import hf_hub_url

# outfolder = where to put metadata
# start = start index of metadata (to download chunk)
# end = end index of metadata
# dl_dedup_set = Indicate whether you'll download the dedup set again (2GB)

def download_hf_metadata(outfolder,start=0,end=2313,dl_dedup_set=True):
    os.makedirs(f'{outfolder}/metadata/',exist_ok=True)

    if dl_dedup_set:
        print('downloading dedup set...')
        url = f'https://huggingface.co/datasets/fraisdufour/snip-dedup/resolve/main/is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy'
        response = requests.get(url)
        # np.save('is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy',response.content)
        # hf_hub_url(repo_id="fraisdufour/snip-dedup", filename="is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy")
        open(f'is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy', "wb").write(response.content)
        
    is_dup_all = np.load('is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy').ravel()
    abs_ind = 0
    for n in range(start,end):
        print(f'downloading metadata file {n}/{end}')
        url = f'https://huggingface.co/datasets/laion/laion2b-en-vit-h-14-embeddings/resolve/main/metadata/metadata_{n:04d}.parquet'
        response = requests.get(url)
        open(f'{outfolder}/metadata/metadata_{n:04d}.parquet', "wb").write(response.content)
        
        # perform the deduplication
        md = pd.read_parquet(f'{outfolder}/metadata/metadata_{n:04d}.parquet')
        non_dup_chunk = is_dup_all[abs_ind:abs_ind+len(md.index)]
        
        # take only non-dupped (uniques)
        non_dup_chunk = np.logical_not(non_dup_chunk)
        
        # make sure there is at least one unique
        non_dup_chunk[0] = True
        
        md = md[non_dup_chunk]
        
        # overwrite metadata
        md.to_parquet(f'{outfolder}/metadata/metadata_{n:04d}.parquet')
        abs_ind+=len(md.index)
    
    
if __name__ == "__main__":
    fire.Fire(download_hf_metadata)
