# Build your own SNIP index from CLIP features

This tutorial aims at being a friendly tour around the `snip` commands enabling building an index.
For this tutorial, we use the `laion-2b-en-vit-l-14` dataset.

Let's start by creating a dedicated virtual environment for this tutorial.

```sh
# Create and activate a virtual environment
mkdir snip_index_tuto
cd snip_index_tuto
python -m venv snip_tuto_venv
source snip_tuto_venv/bin/activate # adapt to your OS/shell
```

Let's continue with an installation of `snip` with the `snip-dedup` package.

```sh
# Install snip
pip install snip-dedup
snip --help
```

Alright, now we need to download the pre-required files for this tutorial:

- The `laion-2b-en-vit-l-14` CLIP embeddings
- The SNIP corresponding model
- The SNIP base index for that model

```sh
# Create directory structure for required files to download
mkdir laion-2b-en-vit-l-14
cd laion-2b-en-vit-l-14
mkdir snip_models
mkdir clip_feats

# Download the CLIP features
cd clip_feats
for i in $(seq -f "%04g" 0 1); do curl -fLO "https://huggingface.co/datasets/laion/laion2b-en-vit-l-14-embeddings/resolve/main/img_emb/img_emb_$i.npy"; done

# Download the SNIP model and base index
cd ../snip_models
curl -fLO https://huggingface.co/datasets/fraisdufour/snip-dedup/resolve/main/models/snip_vitl14_128_deep.pth
curl -fLO https://huggingface.co/datasets/fraisdufour/snip-dedup/resolve/main/index/snip_vitl14_deep_IVFPQ_M4_base.index
```

We are now ready to use the `snip` commands.
We start by compressing the CLIP features with SNIP.

```sh
# Compress CLIP features with SNIP
cd ..
snip compress --help # display the help for the snip compress command
snip compress \
  --snip_model_path snip_models/snip_vitl14_128_deep.pth \
  --parts 0:2 \
  --clip_feats clip_feats/img_emb_{part:04d}.npy \
  --snip_feats_out snip_feats/{part:04d}.npy
```

Finally, after compressing with SNIP, we can build our index.
Since the index is much smaller than the features, we can group multiple parts in each index shard.
In this example, we group them by 2 with `--shard_size 2`

```sh
# Build the index for the SNIP features
snip index --help # display the help for the snip index command
snip index \
  --parts 0:2 \
  --snip_feats snip_feats/{part:04d}.npy \
  --snip_base_index_path snip_models/snip_vitl14_deep_IVFPQ_M4_base.index \
  --index_outdir snip_index \
  --shard_size 2
# will build file snip_index/0000_0001.index
```

That's it!
You now have a sharded compressed SNIP index for the laion2b-vit-l-14 model.
