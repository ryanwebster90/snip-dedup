# snip-dedup

** Disclaimer** 
We are still verifying this result.


SNIP is a very compact index (25GB) that has found roughly half a billion duplicates on the LAION-2B-en dataset (or roughly 1/4 the images). In addition, we release this index for public use and exploration of the LAION datasets. Soon we will release tools to train your own SNIP indices with your own packages

## Setup

Required packages are torch, faiss, numpy and fire.

pip install -r requirements.txt

**
We assume you have all the metadata for laion-2b-en vith14 and some of the features for verification. 

You will need to download the index, cumulative size file, compression model (this feeds into the index) and some other stuff. 









