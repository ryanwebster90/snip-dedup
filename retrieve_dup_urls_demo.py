import faiss
import numpy as np
import glob
import time
import torch
import fire
def abs_ind_to_feat_file(abs_ind, cum_sz, feat_files=None):
    inds = np.argwhere(abs_ind - cum_sz >= 0)
    last_ind = inds[-1].item()
    ind_offset = cum_sz[last_ind]
    local_ind = abs_ind - ind_offset
    if feat_files is not None:
        ff = feat_files[last_ind]
    else:
        ff=None
    return ff,last_ind,local_ind

def get_cum_sz(feat_files):
    cum_sz = [0]
    for feat in feat_files:
        cum_sz += [cum_sz[-1] + np.load(feat,mmap_mode='r').shape[0]]
    cum_sz = np.array(cum_sz).astype('int')
    return cum_sz

def get_emb(ff,local_ind):
    return np.load(ff,mmap_mode='r')[local_ind,:]

def retrieve_duplicate_urls(feats_path, metadata_path,net_path,index_path='mlp_1024_128_gelu_snn_2layer_notext_l2b_vith14_merged.index',cum_sz_file='cum_sz_feats.npy',dup_file = 'is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy'):

    index =  faiss.read_index(index_path)
    index.nprobe = 1
    print('index loaded')

    net = torch.load(net_path).eval().cuda()

    import pandas as pd
    feat_files = sorted(glob.glob(feats_path + '*npy'))
    md = sorted(glob.glob(metadata_path + '*.parquet'))

    is_dup_all = np.load(dup_file)
    cum_sz = get_cum_sz(feat_files)

    n_eval = 1000
    inds = np.argwhere(is_dup_all).ravel()

    r_sample = np.random.randint(0,inds.shape[0], (n_eval,))
    inds = inds[r_sample]
    md_text = open('duplicate_url_pairs.txt','a+')

    thresh_raw = 1e-1
    all_tf = np.full( (n_eval,),False,dtype=bool) 
    for ii,k in enumerate(inds):
        ff,li,lci =  abs_ind_to_feat_file(k,cum_sz,feat_files)
        if li < len(md):
            try:
                # certain metadata entries throw errors, not sure why
                url = list(pd.read_parquet(md[li])["url"])[lci]
            except Exception:
                url = None
        else:
            # note this won't happen if you have all the metadata
            continue
            
        raw_feat = get_emb(ff,lci).reshape(1,-1)

        with torch.no_grad():
            feat_snip = net(torch.from_numpy(raw_feat).float().cuda()).cpu().numpy()

        d,i = index.search(feat_snip,6)
        nn = i[0,1]
        if nn == k:
            print('same nn retrieved, skipping...')
            # Note, this does not effect our de-dup precision
            # but just an artifact of bitwise duplicates, will be fixed later
            
        # only fetch for metadata, if you have all feature file syou can enable gt computation
        ff,li,lci = abs_ind_to_feat_file(nn, cum_sz, None)

        # if you have all the feats, go ahead and compute "ground truth"
        # was used for our precision calculation (see paper)
        # nn_feat = get_emb(ff,lci)
        # mse = ((raw_feat - nn_feat)**2).sum()

        # is_dup = 'gt dup' if mse < thresh_raw else 'nondup'
        if li < len(md):
            try:
                if url is not None:
                    # md_text.write(is_dup + f'\n')
                    url1 = list(pd.read_parquet(md[li])["url"])[lci]
                    if url1 is not None:
                        md_text.write(url + '\n')
                        md_text.write(url1 + '\n')
                        md_text.write('\n')
            except Exception:
                print('failed to parquet')
                

if __name__ == "__main__":
    fire.Fire(retrieve_duplicate_urls)
























