"""snip compress"""
import requests
import os
import os.path
import fire
import numpy as np
import pandas as pd
import glob
import torch

# compute features over chunks
@torch.no_grad()
def compute_feats_for_chunk(net,chunk,batch_size=256):
    feats = []
    for b in range(0,chunk.shape[0],batch_size):
        end_ind = min(b + batch_size,chunk.shape[0])
        batch = chunk[b:end_ind,:]
        feats += [net(torch.from_numpy(batch).float().cuda()).cpu().numpy()]
    feats = np.concatenate(feats,axis=0)
    return feats

def snip_compress(feats_folder, snip_feats_folder='snip_feats/', model_folder='models/', snip_model='snip_vitl14_128_deep',download_models=True,feats_start=0,feats_end=-1):
    """Compress frozen clip features with SNIP

    Parameters
    ----------
    feats_folder : str
        folder with .npy clip features
    model_folder : str
        folder with snip models
    snip_model : str, optional
        Snip model artifact which compresses feats (see available models at fraisdufour/snip-dedup on hf)
    snip_index : str, optional
        snip index built on top of compressed feats
    download_models : bool, optional
        download the model and index
    """
    
    if download_models:
        print("downloading snip artifact...")
        os.makedirs(model_folder,exist_ok=True)
        # download the snip net
        url = f"https://huggingface.co/datasets/fraisdufour/snip-dedup/resolve/main/models/{snip_model}.pth"
        response = requests.get(url)
        open(os.path.join(model_folder,f'{snip_model}.pth'), "wb").write(response.content)
    #TODO: load model
    
    # load snip net
    net = torch.load(os.path.join(model_folder,f'{snip_model}.pth')).eval().cuda()
    
    save_dir = f'{snip_feats_folder}{snip_model}/'
    os.makedirs(save_dir,exist_ok=True)
    
    #subsample feat files
    feat_files = sorted(glob.glob(feats_folder + '/*.npy'))
    feats_end = len(feat_files) if feats_end < 0 else feats_end
    feat_files = feat_files[feats_start:feats_end]
    
    # make snip feats folder
    os.makedirs(snip_feats_folder,exist_ok=True)
    
    for ffi,ff in enumerate(feat_files):
        print(f'computing snip feat {ffi+feats_start}/{len(feat_files)+feats_start}')
        # this is normally the bottleneck
        feat_chunk = np.load(ff)
        # compute snip feats
        snip_feats = compute_feats_for_chunk(net,feat_chunk)
        np.save(f'{save_dir}{ffi+feats_start:06d}.npy',snip_feats)

if __name__ == "__main__":
    fire.Fire(snip_compress)
