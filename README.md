# snip-dedup

** Disclaimer** 
Due to the scale of the dataset, there is still a chance our analysis is false. Use at your own risk. Help for better de-duiplication (higher acc, higher recall) is very much appreciated.

SNIP is a very compact index (25GB) that has found roughly half a billion duplicates on the LAION-2B-en dataset (or roughly 1/4 the images). In addition, we release this index for public use and exploration of the LAION datasets. Soon we will release tools to train your own SNIP indices! More soon tm...

You may find the following necessary files here:
[Binary array of De-duplicated Images](https://drive.google.com/file/d/1RYDylZKaPyaVs5YNwIrGqHU2BewdFwxY/view?usp=sharing)
[SNIP index](https://drive.google.com/file/d/1RYDylZKaPyaVs5YNwIrGqHU2BewdFwxY/view?usp=sharing)



## Setup

Required packages are torch, faiss, numpy and fire.

pip install -r requirements.txt

**
We assume you have all the metadata for laion-2b-en vith14 and some of the features for verification. You may download them [here]()

You will need to download the index, cumulative size file, compression model (this feeds into the index) and some other stuff. 









