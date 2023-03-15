"""snip index"""

import sys
import os
import os.path
import fire
import numpy as np
import faiss

from . import _cli_helper


def snip_index(
    parts="0:2",
    snip_feats="snip_feats/{part:04d}.npy",
    snip_base_index_path="snip_models/snip_vitl14_deep_IVFPQ_M4_base.index",
    index_outdir="snip_index",
    shard_size=1,
):
    """Build a sharded index from SNIP compressed features

    Parameters
    ----------
    parts : str
        Parts to index, using a slice notation, such as 0:2 or 14:42
    snip_feats : str
        Pattern referencing the path to the SNIP features parts.
        You are expected to use the "part" variable with formatting options, such as "{part:03d}.npy" which will be replaced by "001.npy" when part==1.
    snip_base_index_path : str
        Path to the base index, might be something like: snip_models/snip_vitl14_deep_IVFPQ_M4_base.index
    index_outdir : str
        Directory where the computed index parts will be saved.
    shard_size : int
        Number of SNIP parts to group per index shard.
        Since the index is much smaller than the features, we can pack many feature parts in a single index shard.
    """
    # Check that the path for the base index passed as argument is valid
    if not os.path.isfile(snip_base_index_path):
        sys.exit(
            f'The base index file "{snip_base_index_path}" does not exist or is not readable.'
        )

    # Check that the parts argument is correct
    start_part, end_part = _cli_helper.validate_parts(parts)

    # Check that the SNIP features exist
    _cli_helper.validate_part_format(snip_feats)

    # Check that the starting SNIP feature part is a multiple of the index shard size.
    # Otherwise, it's probably an off-by-one mistake
    if start_part % shard_size != 0:
        sys.exit(
            f"WARNING: your starting SNIP part ({start_part}) is not a multiple of your packing argument ({shard_size}). You might be doing a mistake so please double check."
        )

    # TODO: add option for cpu (will be quite slow however)
    res = faiss.StandardGpuResources()

    # Create the output directory for computed index shards
    os.makedirs(index_outdir, exist_ok=True)

    # Group parts into shards
    parts_range = list(range(start_part, end_part))
    grouped_parts = [
        parts_range[i : i + shard_size] for i in range(0, len(parts_range), shard_size)
    ]

    # Load SNIP base index
    base_index = faiss.read_index(snip_base_index_path)

    # Compute an index for each SNIP parts group
    for parts in grouped_parts:
        index = faiss.index_cpu_to_gpu(res, 0, base_index)
        for snip_part_id in parts:
            print(f"Indexing SNIP part {snip_part_id} ...")
            snip_part = np.load(snip_feats.format(part=snip_part_id))
            index.add(snip_part)
        group_str = "_".join([f"{id:04d}" for id in parts])
        print(f"Writing index for parts {parts} ...")
        faiss.write_index(
            faiss.index_gpu_to_cpu(index),
            os.path.join(index_outdir, f"{group_str}.index"),
        )


if __name__ == "__main__":
    fire.Fire(snip_index)
