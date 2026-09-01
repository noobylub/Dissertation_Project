# Emotions Across Languages in Large Language Models

This readme was aided with AI generation tool. 
Code and experimental resources for the MSc dissertation:

**Emotions Across Languages in Large Language Models:  
A Cross-Lingual Emotional Representation Study on Large Language Models**

University of Manchester, 2026.

This project investigates how emotions are represented internally in Large Language Models (LLMs) across **English and Indonesian**, and whether these representations can be used to steer generated text towards particular emotions.

The experiments combine:

- layer-wise linear probing;
- cross-lingual probe transfer;
- hidden-state extraction;
- emotion steering-vector extraction;
- cosine-similarity analysis; and
- activation steering with Indonesian text generation.

The five target emotions are:

```text
anger
fear
happiness
love
sadness
```

A sixth **neutral** category is included for probing and contrastive steering-vector construction.

---

## Research Questions

The project investigates:

1. To what extent is emotion-related information linearly represented in English and Indonesian hidden states?
2. To what extent are emotion representations shared between English and Indonesian?
3. Can emotion vectors derived from English and Indonesian text steer model outputs towards the intended emotion?

---

# Models

Two 8B Llama models are used:

### Base model

```text
meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Indonesian-adapted model

```text
Sahabat-AI/llama3-8b-cpt-sahabatai-v1-instruct
```

Both use the same underlying 32-layer Llama architecture with 4096-dimensional hidden states.

Models are loaded using Hugging Face Transformers with 8-bit `bitsandbytes` quantisation.

---

# Repository Structure

```text
Dissertation_Project/
│
├── activation_steering.ipynb
│   Main notebook for:
│   - loading models and datasets
│   - extracting hidden states
│   - constructing steering vectors
│   - activation steering
│   - generating responses
│
├── hidden_state_analysis.ipynb
│   Main notebook for:
│   - same-language probing
│   - cross-lingual probing
│   - layer-wise analysis
│   - cosine-similarity analysis
│
├── model_code/
│   ├── data_setup.py
│   ├── generate.py
│   ├── hidden_state_analysis.py
│   └── steering_extraction.py
│
├── resources/
│   ├── en_emotion/
│   │   └── goemotions_2.csv
│   │
│   ├── id_emotion/
│   │   ├── AngerData.csv
│   │   ├── FearData.csv
│   │   ├── JoyData.csv
│   │   ├── LoveData.csv
│   │   ├── NeutralData.csv
│   │   └── SadData.csv
│   │
│   ├── neutral_prompts/
│   │   └── prompt_neutral.py
│   │
│   └── saved_vectors/
│       ├── English Vectors/
│       ├── Indonesian Vectors/
│       ├── English Vectors LLAMA ID/
│       └── Indonesian Vectors LLAMA ID/
│
├── outputs/
│   Generated responses from steering experiments.
│
├── images/
│   Figures and visualisations.
│
├── no_need/
│   Older exploratory notebooks. These are not required for reproduction.
│
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the repository

The repository contains large `.pt` files, so Git LFS is recommended.

```bash
git lfs install

git clone https://github.com/noobylub/Dissertation_Project.git

cd Dissertation_Project

git lfs pull
```

---

## 2. Create a Python environment

The original experiments used **Python 3.12.3**.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

A CUDA-capable NVIDIA GPU is strongly recommended.

Check that PyTorch can access CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Ideally this should return:

```text
True
```

---

# Hugging Face Setup

The models are downloaded from Hugging Face.

You will need access to:

```text
meta-llama/Meta-Llama-3.1-8B-Instruct
```

Create a Hugging Face access token and place it in a `.env` file in the repository root:

```text
HF_TOKEN=your_huggingface_token_here
```

The project reads this token automatically when loading the model.

Do not commit your `.env` file or Hugging Face token.

---

# Running the Project

Always launch Jupyter from the repository root:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

The notebooks use relative paths such as:

```text
resources/
outputs/
```

so running them from another directory may cause file-path errors.

There are two ways to reproduce the project.

---

# Option 1 — Use the Existing Saved Vectors

This is the easiest option.

Use this if you want to:

- run the probing experiments;
- reproduce cosine-similarity analysis;
- generate new steered responses; or
- inspect the existing representations.

Pre-computed representations are stored under:

```text
resources/saved_vectors/
```

The folders correspond to:

| Folder | Model | Vector language |
|---|---|---|
| `English Vectors` | Base Llama | English |
| `Indonesian Vectors` | Base Llama | Indonesian |
| `English Vectors LLAMA ID` | Indonesian Llama | English |
| `Indonesian Vectors LLAMA ID` | Indonesian Llama | Indonesian |

For example:

```python
import torch

english_vectors = torch.load(
    "resources/saved_vectors/English Vectors/steering_vectors.pt"
)
```

This allows you to skip the expensive hidden-state extraction stage.

For steering, open:

```text
activation_steering.ipynb
```

For probing and representation analysis, open:

```text
hidden_state_analysis.ipynb
```

---

# Option 2 — Reproduce Everything From Scratch

For a complete reproduction, follow this order:

```text
1. Load English and Indonesian emotion datasets
                    ↓
2. Load the Llama model
                    ↓
3. Extract hidden states
                    ↓
4. Mean-pool token representations
                    ↓
5. Construct emotion steering vectors
                    ↓
6. Save vectors
                    ↓
7. Run activation steering
                    ↓
8. Generate Indonesian responses
                    ↓
9. Run same-language probes
                    ↓
10. Run cross-lingual probes
                    ↓
11. Run cosine-similarity analysis
```

Start with:

```text
activation_steering.ipynb
```

and then run:

```text
hidden_state_analysis.ipynb
```

---

# Dataset Setup

The English data come from **GoEmotions**.

```text
resources/en_emotion/goemotions_2.csv
```

The Indonesian data come from the **Emotion Dataset of Indonesian Public Opinion**.

```text
resources/id_emotion/
```

Six categories are used:

```text
anger
fear
happiness
love
sadness
neutral
```

The final experiment uses:

```text
400 examples per category
6 categories
2,400 examples per language
```

The first **200 examples per category** are used for steering-vector extraction.

All **400 examples per category** are used for probing.

Example:

```python
anger_en, happiness_en, sadness_en, love_en, fear_en, neutral_en = \
    setup.ENEmotionsSetup(
        examples_take=400,
        min_chars=20,
        goemotions_path="resources/en_emotion/goemotions_2.csv"
    )
```

---

# Hidden-State Extraction

Each text is passed through the frozen language model with hidden-state output enabled.

Token representations are mean-pooled at each transformer layer:

```text
token hidden states
        ↓
mean pooling
        ↓
one sentence representation
```

Each sentence therefore produces representations across:

```text
32 layers × 4096 dimensions
```

These representations are used for both probing and steering-vector construction.

---

# Steering-Vector Extraction

For each emotion, a contrastive steering vector is calculated as:

```text
target emotion mean
        -
mean representation of non-target emotions
        =
steering vector
```

For example:

```text
anger vector
=
mean anger representation
-
mean non-anger representation
```

The calculation is performed independently for every transformer layer.

Before steering, vectors are normalised to unit Euclidean norm.

---

# Activation Steering

During inference, the steering vector is added to the model activation:

```text
h_steered = h + λv
```

where:

```text
h = original hidden activation
v = emotion steering vector
λ = steering strength
```

The final experiment applies steering at:

```python
target_layers = [10, 11, 18, 19, 28, 29]
```

using:

```python
steering_strengths = [1.15, 1.30]
```

Generation uses:

```python
max_new_tokens = 250
do_sample = False
```

`do_sample=False` gives greedy decoding and reduces random variation between outputs.

---

# Steering Conditions

The experiment compares four conditions:

| Condition | Model | Vector source |
|---|---|---|
| EN–EN | Base Llama | English |
| EN–ID | Base Llama | Indonesian |
| ID–EN | Indonesian Llama | English |
| ID–ID | Indonesian Llama | Indonesian |

All final responses are generated in Indonesian.

The evaluation uses the same 8 Indonesian prompts across conditions.

---

# Linear Probing

Probe analysis is performed in:

```text
hidden_state_analysis.ipynb
```

A separate multiclass linear probe is trained for each transformer layer.

The probe predicts:

```text
anger
fear
happiness
love
sadness
neutral
```

The final configuration uses:

```text
8-fold stratified cross-validation
25 epochs
batch size = 50
learning rate = 0.001
Adam optimiser
cross-entropy loss
seed = 42
```

Because the six classes are balanced, random chance is approximately:

```text
16.7%
```

---

# Cross-Lingual Probing

Four probe conditions are evaluated:

```text
English → English
Indonesian → Indonesian
English → Indonesian
Indonesian → English
```

The cross-language conditions test whether a probe trained on representations from one language can classify representations from the other.

Probe performance is evaluated across all 32 transformer layers.

---

# Cosine-Similarity Analysis

Emotion steering vectors are also compared using cosine similarity.

The analysis uses the five target emotions:

```text
anger
fear
happiness
love
sadness
```

and produces pairwise similarity matrices.

Representative layers used in the dissertation are:

```text
5
16
28
```

These provide examples from relatively early, middle, and later parts of the model.

---

# Final Dissertation Configuration

For exact reproduction of the reported experiments, use:

```python
NUM_EXAMPLES_PER_EMOTION = 400
NUM_STEERING_EXAMPLES = 200

TARGET_LAYERS = [10, 11, 18, 19, 28, 29]

STEERING_STRENGTHS = [1.15, 1.30]

MAX_NEW_TOKENS = 250
DO_SAMPLE = False

PROBE_FOLDS = 8
PROBE_EPOCHS = 25
PROBE_BATCH_SIZE = 50
PROBE_LEARNING_RATE = 1e-3

RANDOM_SEED = 42
```

Some notebook cells were used for exploratory experiments and may contain different steering strengths or settings.

When reproducing the final dissertation results, use the configuration above.

---

# Expected Workflow

For a full reproduction:

```text
git clone + git lfs pull
        ↓
install requirements
        ↓
create .env with HF_TOKEN
        ↓
launch Jupyter
        ↓
run activation_steering.ipynb
        ↓
extract/save representations
        ↓
generate steering outputs
        ↓
run hidden_state_analysis.ipynb
        ↓
run probing
        ↓
run cross-lingual analysis
        ↓
run cosine-similarity analysis
```

If the saved `.pt` files are already available, hidden-state extraction can be skipped.

---

# Generated Outputs

Generated steering responses are stored in:

```text
outputs/
```

The output files correspond to the four model/vector conditions.

Figures generated during analysis are stored under:

```text
images/
```

---

# Troubleshooting

### Hugging Face 401 / 403 error

Check that:

- your Hugging Face account has access to the Llama model;
- your token is valid; and
- `.env` contains:

```text
HF_TOKEN=...
```

Restart the notebook kernel after changing `.env`.

---

### CUDA out of memory

Try:

- running one model at a time;
- restarting the kernel before switching models;
- closing other GPU processes; or
- using a GPU with more memory.

You can inspect GPU usage with:

```bash
nvidia-smi
```

---

### `.pt` files are missing or extremely small

Run:

```bash
git lfs pull
```

The repository uses Git LFS for large saved representation files.

---

### `FileNotFoundError`

Make sure Jupyter was launched from the repository root.

Check:

```python
import os
print(os.getcwd())
```

The path should end in:

```text
Dissertation_Project
```

---

# Reproducibility Notes

The notebooks contain some exploratory code from earlier stages of the project.

For the final reported experiments, use the values under:

```text
Final Dissertation Configuration
```

rather than assuming every historical notebook cell represents the final experimental setup.

The English and Indonesian datasets are also **not parallel translations**. Differences between languages may therefore reflect both linguistic differences and dataset-specific properties.

The language models remain frozen during probing and activation steering. Steering modifies hidden activations only during inference.

---

# Main Code Files

### `model_code/data_setup.py`

Handles:

- dataset loading;
- Hugging Face authentication;
- model loading; and
- 8-bit quantisation.

### `model_code/steering_extraction.py`

Handles:

- hidden-state extraction;
- mean pooling;
- steering-vector construction;
- vector normalisation; and
- forward-hook activation steering.

### `model_code/generate.py`

Handles:

- batch generation;
- multiple steering strengths; and
- saving responses.

### `model_code/hidden_state_analysis.py`

Handles:

- probe training;
- cross-validation;
- cross-lingual evaluation; and
- cosine-similarity analysis.

---

# Data Sources

### English

**GoEmotions**  
Demszky et al. (2020)

### Indonesian

**Emotion Dataset of Indonesian Public Opinion**  
Saputra, Pratama, and Chowanda (2022)


---

# Quick Checklist

```text
[ ] Clone repository
[ ] Run git lfs pull
[ ] Create Python 3.12 environment
[ ] Install requirements
[ ] Obtain Hugging Face model access
[ ] Create .env containing HF_TOKEN
[ ] Confirm CUDA is available
[ ] Launch Jupyter from repository root
[ ] Run activation_steering.ipynb for extraction/generation
[ ] Run hidden_state_analysis.ipynb for probing/analysis
[ ] Use steering strengths 1.15 and 1.30
[ ] Use steering layers 10, 11, 18, 19, 28, 29
```
