# Saved features and experiment artifacts

Both collections have completed embedding caches in private NYU Dataproc/HDFS storage. This Git commit contains summary metrics and portable configuration templates; it does not distribute binary caches, datasets, model weights, full per-column annotations, or raw API logs.

## Cache names

The recorded embedding model was available as `./model`. The original code consequently used these exact cache filenames:

```text
column_name_embeddings_._model.pickle
column_value_embeddings_._model.pickle
column_num_value_embeddings_._model.pickle
```

The name and text-value dictionaries contain embedding vectors; the numeric dictionary contains 27-dimensional feature vectors. Numeric-only tables may have no text-value entry, so text and numeric dictionary table counts need not match.

## Restoring from the existing experiment

The experiment owner can export the completed `NYC-E-artifacts.tar.gz` and `NYC-CG-artifacts.tar.gz` archives from the experiment's private HDFS run directories. Each archive contains `results/<collection>/embeddings/`, together with the corresponding baseline and StraTyper outputs. Keep the collections separate. The final NYC-CG run also saved a standalone `embedding-cache.tar.gz`.

Copy each collection's three cache files into the directory selected by its `embedding_path` configuration. Preserve the model identifier `./model`, table basenames, column names, input data, and original feature settings. The original CLI reads these cached features when it calls `compute_clusters(..., recompute=False)`; API stages can then be run without re-encoding the table values.

Use pickle artifacts only from a trusted source: loading pickle can execute code. Binary checksums were not collected for this public summary; verify any exported artifacts before sharing or reuse. Private storage access and artifact distribution must be arranged with the experiment owner.
