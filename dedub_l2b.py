import pandas as pd
import fire
import numpy as np
import os
import glob
# De-duplicate the LAION-2B-en dataset

def deduplicate_l2b(metadata_path,new_path,dup_file = 'is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy'):
    mdfs = sorted(glob.glob(metadata_path + '*.parquet'))
    is_dup_all = np.load(dup_file).ravel()
    abs_ind = 0
    
    os.makedirs(new_path,exist_ok=True)
    
    for mdi,mdf in enumerate(mdfs):
        
        print(f'de duplicating {mdi}/{len(mdfs)}...')
        
        md = pd.read_parquet(mdf)
        non_dup_chunk = is_dup_all[abs_ind:abs_ind+len(md.index)]
        
        # take only non-dupped (uniques)
        non_dup_chunk = np.logical_not(non_dup_chunk)
        
        print(non_dup_chunk[:10])
        # make sure there is at least one unique
        non_dup_chunk[0] = True
        
        md = md[non_dup_chunk]
        
        out_file = mdf.split('/')[-1]
        
        # write new metadata file
        md.to_parquet(new_path + '/' + out_file)
        
        abs_ind+=len(md.index)

if __name__ == "__main__":
    fire.Fire(deduplicate_l2b)
