# NYC-E / NYC-CG: completed OpenAI-substituted reproduction

The original StraTyper code completed discovery, annotation, and built-in evaluation for both collections. The direct LLM baseline achieved higher Precision and Hits on the original evaluator's comparison subset in this run.

This is **one run with approved OpenAI model substitutions**, not an exact reproduction of the paper's model configuration. The experiment used source commit `36144561f6334143bc24b58794c86687a2411c60`; its source files, prompts, feature construction, and evaluator were unchanged. This directory adds results and configuration templates only.

Completed: **2026-11-23 03:56:45 UTC** (2026-11-22 22:56:45 America/New_York). Final YARN state: **FINISHED / SUCCEEDED**.

## Results

| Collection | Method | Precision | Hits | Description similarity |
| --- | --- | ---: | ---: | ---: |
| NYC-E | StraTyper | 0.582 | 0.660 | 0.568 |
| NYC-E | StraTyper_LLM (direct baseline) | 0.833 | 0.967 | 0.592 |
| NYC-CG | StraTyper | 0.592 | 0.776 | 0.494 |
| NYC-CG | StraTyper_LLM (direct baseline) | 0.835 | 0.979 | 0.554 |

| Collection | Tables | Columns | Columns with nonempty StraTyper types | Discovered types |
| --- | ---: | ---: | ---: | ---: |
| NYC-E | 373 | 3,409 | 2,896 (84.95%) | 334 |
| NYC-CG | 331 | 3,071 | 1,927 (62.75%) | 273 |

**Evaluation scope:** the original evaluator compares methods on columns annotated by StraTyper. The baseline rows above are therefore evaluated on the same subset, not on all columns. Description similarity retains the original evaluator's reported label, `BERTScore Description Similarity`. These are reported single-run values, with no multi-seed uncertainty estimate.

Full reported numeric/textual breakdowns and available token counts are in [results.json](results.json) and [results.csv](results.csv). The original code does not report judge token usage; token counts are incomplete for total API spend.

## Models and parameters

- Discovery and direct baseline: `gpt-4.1-2025-04-14`.
- Closed-set annotation and LLM judge: `gpt-4.1-mini-2025-04-14`.
- Embeddings: `sentence-transformers/paraphrase-mpnet-base-v2`, localized as `./model`.
- Name/value clustering thresholds: `[0.99, 0.9, 0.75]`; minimum cluster size: 2.
- Original sampling: `length`; context: `most_frequent_5`; 10 value samples; 3 CCTA context rows; refinement disabled.
- Original value batch size: 256; maximum sequence length: 512; numeric features: histogram 20 + magnitude 7 = 27 dimensions.

Key observed environment versions and resource settings are in [environment.json](environment.json).

## Running with the original CLIs

Obtain the datasets from the link in the repository's root README. Extract each collection's CSV tables and corresponding metadata into separate directories. Set up the original requirements and provide your own `OPENAI_API_KEY` through the environment or a local untracked credential file.

Edit all `/absolute/path/to/experiment/...` placeholders in the four [config templates](config). Run from the repository root with the embedding model available as `./model`. For NYC-E:

```bash
python src/llm_inference.py --config experiments/nyc-openai-2026-11-22/config/NYC-E-baseline.yaml --store_path /absolute/path/to/experiment/NYC-E/baseline
python src/stratyper.py --config experiments/nyc-openai-2026-11-22/config/NYC-E-stratyper.yaml --print --evaluate
```

Repeat using the NYC-CG templates and paths. Running the baseline or StraTyper's API stages can incur API charges. API outputs may vary across runs. To reuse saved features, follow [ARTIFACTS.md](ARTIFACTS.md).

## Execution and recovery

Computation ran on CPU, with 4 threads and a 32 GiB embedding memory budget. NYC-CG's original value-embedding function was called one table at a time by an external operational wrapper; completed table dictionaries were persisted to HDFS and merged in input directory order under the original cache filenames. Actual computation ran inside a Spark executor task. Clean application boundaries after approximately 90 minutes limited the scope of interrupted work.

Two infrastructure interruptions in the final NYC-CG campaign were automatically recovered from saved artifacts. Earlier attempts included additional forced termination and an API credit failure. Successful stages and completed table features were reused; no row sampling, feature truncation, prompt edit, or algorithm change was introduced for recovery. Operational helpers and private storage locations are not bundled in this results commit.

This directory records the observed run. It does not claim bitwise deterministic reproduction or independent verification against the paper's published numerical results.
