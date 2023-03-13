"""snip compress"""
import requests
import os
import os.path
import fire
import numpy as np
import glob
import faiss

def snip_index(feats_folder, model_folder='models/', snip_index='snip_vitl14_deep_IVFPQ_M4_base',download_models=True,feats_start=0,feats_end=-1,shard_folder='none',shard_every=1):
    """Compress frozen clip features with SNIP

    Parameters
    ----------
    feats_folder : str
        folder with .npy snip features
    model_folder : str
        folder with snip models
    snip_index : str, optional
        snip index built on top of compressed feats
    download_models : bool, optional
        download the model and index
    shard_folder : str, optional
        output folder for the computed shards
    shard_every : int, optional
        How often to save shards
    """
    
    if download_models:
        print("downloading snip index...")
        os.makedirs(model_folder,exist_ok=True)
        url = f"https://huggingface.co/datasets/fraisdufour/snip-dedup/resolve/main/index/{snip_index}.index"
        response = requests.get(url)
        open(os.path.join(model_folder,f'{snip_index}.index'), "wb").write(response.content)
    # TODO: just load models
    
    # load snip net
    base_index_file = os.path.join(model_folder, f'{snip_index}.index')
    index = faiss.read_index(base_index_file)
    
    # TODO: add option for cpu (will be quite slow however)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res,0,index)
    
    #subsample feat files
    feat_files = sorted(glob.glob(feats_folder + '/*.npy'))
    feats_end = len(feat_files) if feats_end < 0 else feats_end
    feat_files = feat_files[feats_start:feats_end]
    
    # make folder to save index shards
    if shard_folder == 'none':
        shard_folder = os.path.join(model_folder,f'{snip_index}_shards/')
    
    # make snip feats folder
    os.makedirs(f'{shard_folder}',exist_ok=True)
    
    for ffi,ff in enumerate(feat_files):
        print(f'adding snip feat {ffi+feats_start}/{len(feat_files)+feats_start} to index')
        # this is normally the bottleneck
        feat_chunk = np.load(ff)
        
        # add to index (todo: benchmark and progress)
        index.add(feat_chunk)
        
        if (ffi+1)%shard_every==0:
            # write index shard, name according to the feature files within the shard
            faiss.write_index(faiss.index_gpu_to_cpu(index), f'{shard_folder}{ffi + feats_start - shard_every + 1}_{ffi + feats_start + 1}_shard.index')
            
            # re-load the base index for the new shard
            index = faiss.read_index(base_index_file)
            
if __name__ == "__main__":
    fire.Fire(snip_index)
