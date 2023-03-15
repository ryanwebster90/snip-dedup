"""snip index"""

import sys
import os
import os.path
import fire
import numpy as np
import faiss

from . import _cli_helper


def snip_index(
    snip_shards="0:2",
    snip_feats="snip_feats/{shard:04d}.npy",
    snip_base_index_path="snip_models/snip_vitl14_deep_IVFPQ_M4_base.index",
    index_outdir="snip_index",
    index_shard_packing=1,
):
    """Build a sharded index from SNIP compressed features

    Parameters
    ----------
    snip_shards : str
        Shards to index, using a slice notation, such as 0:2 or 14:42
    snip_feats : str
        Pattern referencing the path to the SNIP features shards.
        You are expected to use the "shard" variable with formatting options, such as "{shard:03d}.npy" which will be replaced by "001.npy" when shard==1.
    snip_base_index_path : str
        Path to the base index, might be something like: snip_models/snip_vitl14_deep_IVFPQ_M4_base.index
    index_outdir : str
        Directory where the computed index shards will be saved.
    index_shard_packing : int
        Number of SNIP shards to group per index shard.
        Since the index is much smaller than the features, we can pack many feature shards in a single index shard.
    """
    # Check that the path for the base index passed as argument is valid
    if not os.path.isfile(snip_base_index_path):
        sys.exit(
            f'The base index file "{snip_base_index_path}" does not exist or is not readable.'
        )

    # Check that the shards argument is correct
    start_shard, end_shard = _cli_helper.validate_shards(snip_shards)

    # Check that the SNIP features exist
    _cli_helper.validate_shard_format(snip_feats)

    # Check that the starting SNIP feature shard is a multiple of the index shard packing.
    # Otherwise, it's probably an off-by-one mistake
    if start_shard % index_shard_packing != 0:
        sys.exit(
            f"WARNING: your starting SNIP shard ({start_shard}) is not a multiple of your packing argument ({index_shard_packing}). You might be doing a mistake so please double check."
        )

    # TODO: add option for cpu (will be quite slow however)
    res = faiss.StandardGpuResources()

    # Create the output directory for computed index shards
    os.makedirs(index_outdir, exist_ok=True)

    # Batch shards into groups
    snip_shards = list(range(start_shard, end_shard))
    batched_snip_shards = [
        snip_shards[i : i + index_shard_packing]
        for i in range(0, len(snip_shards), index_shard_packing)
    ]

    # Load SNIP base index
    base_index = faiss.read_index(snip_base_index_path)

    # Compute an index for each SNIP shards batch
    for shards_batch in batched_snip_shards:
        index = faiss.index_cpu_to_gpu(res, 0, base_index)
        for snip_shard_id in shards_batch:
            print(f"Indexing SNIP shard {snip_shard_id} ...")
            snip_shard = np.load(snip_feats.format(shard=snip_shard_id))
            index.add(snip_shard)
        batch_str = "_".join([f"{id:04d}" for id in shards_batch])
        print(f"Writing index for shards {shards_batch} ...")
        faiss.write_index(
            faiss.index_gpu_to_cpu(index),
            os.path.join(index_outdir, f"{batch_str}.index"),
        )


if __name__ == "__main__":
    fire.Fire(snip_index)
