# StraTyper: Automated Semantic Type Discovery and Multi-Type Annotation for Dataset Collections

> This repository contains the official codebase for the paper **"StraTyper: Automated Semantic Type Discovery and Multi-Type Annotation for Dataset Collections"**.

StraTyper is a cost-effective framework for *Column Type Discovery* and *Column Multi-Type Annotation* across a dataset collection. The framework operates in two phases:

- **Semantic Type Discovery**: First, StraTyper employs clustering and an optimized prompt synthesis to guide an LLM to discover semantic types across the collection.
- **Closed-Set Type Annotation**: After discovering the types, StraTyper propagates them and attempts to annotate remaining columns, by retrieving targeted type candidate sets to use an LLM for annotation.

## 📋 Contents

* [Environment Setup](#environment-setup)
* [Code Structure](#code-structure)
* [Configuration](#configuration)
* [Usage](#usage)
* [Datasets](#datasets)

---

## Environment Setup

### Installation

### Clone the repository
```bash
git clone https://github.com/VIDA-NYU/stratyper.git
cd stratyper
```
### Install dependencies
```bash
conda create -n stratyper python=3.11 -y
conda activate stratyper
pip install -r requirements.txt
```

### Set up API Keys 


This project requires API keys to communicate with LLM providers (OpenAI, Portkey, OpenRouter).

#### Option 1 (`.env`)
* Create the file:


```bash
touch .env

```

* Open `.env` and add your keys:

```text
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...

# Add others as needed

```

#### Option 2

You can also set the API key in your terminal

* For Windows:

```bash
set OPENAI_API_KEY=...
```
* For macOS/Linux:

```bash
export OPENAI_API_KEY=...
```



---

## Code Structure

Here is an overview of the project's file organization:

```bash
|-- src # source code files
	|-- stratyper.py # use to run StraTyper
	|-- llm_inference.py # use to run LLM baselines
|-- config # configuration files for StraTyper and LLM-Baselines
```

## Configuration

We use a YAML configuration file to manage parameters and avoid long command-line arguments.

### Create your local configs

#### LLM Baseline

```bash
cp llm_baseline.example.yaml llm_baseline.yaml

```

#### StraTyper

```bash
cp stratyper.example.yaml stratyper.yaml

```

Edit `.yaml` to match your experiment needs. Example configuration files include explanations of all parameters.

## Usage

### Basic Run

Once your `.env` and `.yaml` configurations are set up, you can run either an LLM baseline or StraTyper.

#### LLM Baseline

```bash
python src/llm_inference.py --config config/llm_baseline.yaml [OPTIONAL --store_path]

```

#### StraTyper

```bash
python src/stratyper.py --config config/stratyper.yaml [OPTIONAL --print --evaluate --restart_eval]

```

#### Supported Arguments

| Flag | Description | Default |
| --- | --- | --- |
| `--store_path` | The path to store annotations from LLM Baseline| `False`|
| `--print` | Print progress | `False` |
| `--evaluate` | Run LLM-judge evaluation | `False` |
| `--restart_eval` | Rerun evaluation | `False` |

#### Remarks

* LLM logs are stored automatically after every run in the specified results path.
* Re-running StraTyper with the same config will cause the method to read stored results. It is possible to run evaluation after running StraTyper without the `--evaluate` flag.

## Datasets

Datasets and corresponding metadata can be found in [Google Drive](https://drive.google.com/file/d/1zQMKkAe9MS4rcYx-_0kLNOIlDPiD7YCT/view?usp=drive_link). Simply download them and specify necessary paths in the `.yaml` configurations.
