<!--
PURPOSE: What scripts/04_mtf.py saves here and why none of it is committed.
WHAT GOES HERE: One patch-embedding tensor per reader as float32 (.npy), its preprocessor_config.json and a record of the fetch. Nothing else.
FILLED IN AT STEP: W3.6.
DO NOT: Commit anything but this README (data/external/** is git-ignored). Download a whole checkpoint here. Edit a saved tensor.
-->

# data/external/weights/

- **What belongs here:** `<reader>__patch_embedding.npy`, the patch-embedding weight of each reader of `configs/readers.yaml` as float32 (lossless from the Hub's BF16); `<reader>__preprocessor_config.json`, the processor's config at the same revision, for `image_std`; `<reader>__record.json`, the repository, revision, tensor name, shard, byte count, SHA-256 and time of the fetch.
- **Where it comes from:** `python scripts/04_mtf.py` (`make mtf`) fetches exactly the tensor's bytes from the Hugging Face Hub at the revision pinned in `configs/readers.yaml`, by HTTP range request, and stops unless their SHA-256 equals the table in the script (build guide 3S.1). A few megabytes in all; never a checkpoint.
- **Committed:** no, only this README; the files are regenerable and `results/mtf/manifest.json` records their hashes, shapes and revisions.
- **Rule:** `scripts/04_mtf.py --offline` reads these files instead of fetching, and checks each tensor's bytes against the table again; a changed file stops it. The `model`-marked test of `tests/test_mtf.py` reads them too.
