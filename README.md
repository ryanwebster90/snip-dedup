# snip-dedup

** Disclaimer** 
We are still verifying this result.


SNIP is a very compact index that can be used to find duplicates on the LAION-2B-en dataset. We've found a large number of duplicates and will release the set to the community soon, so that a "de-duplicated" LAION-2B-en can be downloaded.


## Setup

Required packages are torch, faiss, numpy and fire.

We assume you have all the metadata for laion-2b-en vith14 and some of the features for verification. 

You will need to download the index, cumulative size file, compression model (this feeds into the index) and some other stuff. 









