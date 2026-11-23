# Saved embeddings: download, verify, and reuse

[Download the completed NYC-E and NYC-CG caches (27.5 MB ZIP)](stratyper-nyc-embeddings-2026-11-22.zip?raw=1).

The archive contains six original pickle files: column-name embeddings, text-value embeddings, and numeric feature vectors for each collection. The original experiment source commit is `36144561f6334143bc24b58794c86687a2411c60` and the embedding model is `sentence-transformers/paraphrase-mpnet-base-v2`, localized as `./model`. Dataset contents, model weights, credentials, and raw API logs are not included.

## Verify and extract

From this directory, with the ZIP and `SHA256SUMS` present:

```bash
# macOS
shasum -a 256 -c SHA256SUMS
# Linux alternative: sha256sum -c SHA256SUMS
unzip stratyper-nyc-embeddings-2026-11-22.zip -d embeddings
cd embeddings
shasum -a 256 -c SHA256SUMS
# Linux alternative: sha256sum -c SHA256SUMS
```

The outer ZIP SHA-256 is:

```text
aa3a2b7e3edf9f937a052c4a415ab104100e5c9d296ded5704de0d4398031194
```

The embedded manifest and [embedding-manifest.json](embedding-manifest.json) record each file's byte size, hash, dictionary table/column counts, and vector dimensions. The ZIP hash and all six file hashes were checked against the original server export before publication. Only load pickle artifacts from a trusted source, since unpickling can execute code.

## Reuse with the original CLI

After extraction, set the NYC-E configuration's `embedding_path` to the absolute path of `embeddings/NYC-E`, and the NYC-CG configuration's `embedding_path` to the absolute path of `embeddings/NYC-CG`. Keep these exact cache filenames inside each directory:

```text
column_name_embeddings_._model.pickle
column_value_embeddings_._model.pickle
column_num_value_embeddings_._model.pickle
```

Preserve `models.embedding_model: ./model`, and run with the same embedding model available as `./model` in the working directory. The original code constructs cache filenames from that model-path string. Use the same input CSVs, table basenames, column names, and feature settings as the recorded run.

The original CLI calls `compute_clusters(..., recompute=False)` and reads these existing cache files. Type discovery, annotation, and evaluation can then run without recomputing table embeddings. These caches do not include baseline annotations; configure or generate the comparison baseline separately if needed.

Name and text-value vectors have 768 dimensions. Numeric vectors have 27 dimensions (20 histogram bins plus 7 magnitude features). Numeric-only tables may have no text-value entry, so text and numeric dictionary table counts need not match.

Complete per-column predictions and the full original-generated run archives remain in private NYU Dataproc/HDFS storage.
