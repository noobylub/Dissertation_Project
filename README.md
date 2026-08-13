# Cross-Lingual Emotion Steering in Large Language Models

This repository contains the code, notebooks, datasets, prompts, saved activation vectors, and analysis used for a dissertation project on **cross-lingual emotion representation and activation steering in Large Language Models (LLMs)**.

The project investigates two related questions:

1. **How are emotions represented internally in English and Indonesian?**
2. **Can emotion-related activation directions extracted in one language be used to influence model outputs, and do English- and Indonesian-derived directions behave similarly?**

The experiments focus on six emotion categories:

- Anger
- Happiness
- Sadness
- Fear
- Love
- Neutral

---

## Project Overview

Large Language Models can generate emotionally expressive language, but this does not necessarily mean that emotions are represented in the same way across languages.

This project studies English and Indonesian emotion representations using two complementary approaches:

### 1. Internal representation analysis

Hidden states are extracted from emotional text and analysed using:

- binary linear probes;
- multiclass probes;
- cosine similarity;
- cross-lingual probe transfer;
- similarity-matrix comparison; and
- layer-wise representation analysis.

The aim is to test whether emotion information is linearly recoverable from hidden states and whether the organisation of emotion representations is similar across English and Indonesian.

### 2. Activation steering

For each emotion, a contrastive steering direction is constructed from hidden states and injected into selected transformer layers during generation.

A steering direction for emotion `e` is computed as:

```text
mean hidden state for emotion e
-
mean hidden state across the other emotion categories
```

For example:

```text
anger steering vector
=
mean(anger hidden states)
-
mean(non-anger emotion hidden states)
```

The resulting directions are then used to test whether model responses can be shifted toward a target emotion, and whether vectors derived from English and Indonesian data behave differently.

---

## Research Questions

The repository supports the following dissertation questions:

1. **How are emotions represented internally in each language?**
2. **Do emotion vectors extracted natively from English and Indonesian hidden states align and share similar structure?**
3. **Can vectorised emotion representations be used to steer outputs across languages?**
4. **Do vectorised emotion representations derived from different languages yield similar behavioural effects?**

---

## Repository Structure

```text
Dissertation_Project/
│
├── images/
│   └── Figures and visual outputs used in the analysis
│
├── model_code/
│   ├── __init__.py
│   ├── data_setup.py
│   ├── generate.py
│   ├── hidden_state_analysis.py
│   └── steering_extraction.py
│
├── resources/
│   ├── anger_prompt/
│   ├── fear_prompt/
│   ├── neutral_prompts/
│   ├── en_emotion/
│   ├── id_emotion/
│   ├── saved_vectors/
│   ├── prompt_scenarios.py
│   └── prompt_scenarios_neu.py
│
├── steer_emotions.ipynb
├── vectors_analysis.ipynb
├── just_in_case.ipynb
├── requirements.txt
└── README.md
```

---

## Main Files

### `steer_emotions.ipynb`

This is the main notebook for:

- loading the English and Indonesian emotion datasets;
- constructing the emotion dictionaries;
- extracting hidden states;
- creating steering vectors;
- saving activation vectors; and
- generating responses with activation steering.

If you want to reproduce the steering pipeline, start here.

### `vectors_analysis.ipynb`

This notebook contains the main internal-representation analysis.

It loads previously extracted activation vectors and is used for:

- binary probing;
- multiclass probing;
- cross-lingual probe evaluation;
- layer-wise comparisons;
- cosine-similarity analysis; and
- statistical analysis of representation results.

### `model_code/data_setup.py`

Contains functions for loading the English and Indonesian emotion datasets.

It also contains `modelSetup()`, which loads a Hugging Face causal language model using 8-bit quantisation.

### `model_code/steering_extraction.py`

Contains the main activation-extraction and steering functions, including:

- hidden-state extraction;
- contrastive steering-vector construction;
- vector normalisation;
- forward-hook based activation steering; and
- steered text generation.

### `model_code/generate.py`

Contains utilities for generating responses across:

- multiple prompts; and
- multiple steering strengths.

### `model_code/hidden_state_analysis.py`

Contains utilities used for representation analysis, including:

- binary probes;
- multiclass probes;
- cosine-similarity matrices;
- Mantel-style matrix comparison; and
- helper functions for evaluation.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/noobylub/Dissertation_Project.git
cd Dissertation_Project
```

## 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

The repository currently uses packages including PyTorch, Transformers, Accelerate, bitsandbytes, scikit-learn, NumPy, pandas, SciPy, Matplotlib, and Jupyter.

---

# Hugging Face Setup

The default model loader in `model_code/data_setup.py` uses:

```text
meta-llama/Meta-Llama-3.1-8B-Instruct
```

Because this is a gated Hugging Face model, you must have access to the model and provide a Hugging Face token.

## 1. Create a `.env` file

Create a file called:

```text
.env
```

in the repository root.

Add:

```text
HF_TOKEN=your_huggingface_token_here
```

Do not commit this file. `.env` is already ignored by the repository's `.gitignore`.

## 2. Load the model

The simplest way to use the same setup as the project code is:

```python
import model_code.data_setup as setup

model, tokenizer = setup.modelSetup()
```

To use another compatible Hugging Face causal language model:

```python
model, tokenizer = setup.modelSetup(
    model_name="your-model-name"
)
```

`modelSetup()` currently loads the model using 8-bit quantisation through `bitsandbytes` and `device_map="auto"`.

---

# Running the Project

The easiest way to reproduce the project is to run the notebooks rather than treating the repository as a standalone Python package.

Start Jupyter:

```bash
jupyter notebook
```

Then use:

```text
steer_emotions.ipynb
```

for vector extraction and activation steering, and:

```text
vectors_analysis.ipynb
```

for probing and representation analysis.

---

# Example: Loading the Emotion Data

The steering notebook loads up to 400 examples per emotion from each language.

```python
import model_code.data_setup as setup

anger_en, happiness_en, sadness_en, love_en, fear_en, neutral_en = (
    setup.ENEmotionsSetup(
        examples_take=400,
        min_chars=20,
        goemotions_path="resources/en_emotion/goemotions_2.csv"
    )
)

anger_id, happiness_id, sadness_id, neutral_id, fear_id, love_id = (
    setup.IDEmotionsSetup(
        examples_take=400,
        emotion_dir="resources/id_emotion"
    )
)
```

The return order differs slightly between the English and Indonesian loading functions, so keeping the assignment order shown above is important.

---

# Example: Preparing Steering and Probe Data

The notebook uses the first 200 examples per emotion for steering-vector extraction.

```python
indo_emotion = {
    "anger": anger_id[:200],
    "happiness": happiness_id[:200],
    "sadness": sadness_id[:200],
    "neutral": neutral_id[:200],
    "fear": fear_id[:200],
    "love": love_id[:200],
}

eng_emotion = {
    "anger": anger_en[:200],
    "happiness": happiness_en[:200],
    "sadness": sadness_en[:200],
    "neutral": neutral_en[:200],
    "fear": fear_en[:200],
    "love": love_en[:200],
}
```

A larger set can be retained separately for probing and hidden-state analysis:

```python
indo_emotion_probe = {
    "anger": anger_id[:400],
    "happiness": happiness_id[:400],
    "sadness": sadness_id[:400],
    "neutral": neutral_id[:400],
    "fear": fear_id[:400],
    "love": love_id[:400],
}

eng_emotion_probe = {
    "anger": anger_en[:400],
    "happiness": happiness_en[:400],
    "sadness": sadness_en[:400],
    "neutral": neutral_en[:400],
    "fear": fear_en[:400],
    "love": love_en[:400],
}
```

---

# Example: Extracting Steering Vectors

```python
from model_code.steering_extraction import retrieve_steering_vector

indo_steering_vectors, indo_hidden_states = retrieve_steering_vector(
    model=model,
    tokenizer=tokenizer,
    emotion_sets=indo_emotion,
    name_folder="Indonesian Vectors",
    layer_id=[21]
)

eng_steering_vectors, eng_hidden_states = retrieve_steering_vector(
    model=model,
    tokenizer=tokenizer,
    emotion_sets=eng_emotion,
    name_folder="English Vectors",
    layer_id=[21]
)
```

The function saves steering vectors under:

```text
resources/saved_vectors/<name_folder>/steering_vectors.pt
```

and returns:

```text
steering_vectors, emotion_vectors
```

Each steering vector has one direction per transformer layer, so a target emotion can be accessed with:

```python
eng_steering_vectors["anger"]
indo_steering_vectors["anger"]
```

The `layer_id` argument determines which layer activations are retained in the per-example `emotion_vectors` output for later analysis. The contrastive steering tensors themselves contain directions for all transformer layers.

---

# Example: Normalising a Steering Vector

Before steering, the project can normalise each layer direction to unit norm:

```python
from model_code.steering_extraction import norm_vectors

anger_vector = norm_vectors(
    indo_steering_vectors["anger"]
)
```

This allows steering strength to be applied more consistently across layers.

---

# Example: Generating a Steered Response

```python
from model_code.steering_extraction import generateSteering, norm_vectors

prompt = (
    "Teman dekatmu membatalkan janji penting di menit terakhir "
    "tanpa menjelaskan alasannya. Bagaimana kamu akan merespons?"
)

system_text = "Jawablah sebagai manusia biasa dalam bahasa Indonesia."

steering_vector = norm_vectors(
    indo_steering_vectors["anger"]
)

generated_text = generateSteering(
    user_text=prompt,
    system_text=system_text,
    model=model,
    tokenizer=tokenizer,
    steering_vector=steering_vector,
    steering_strength=1.5,
    target_layers=[21],
    max_new_tokens=150,
    temperature=0.7
)

print(generated_text)
```

`generateSteering()` uses forward hooks to add the selected steering direction to the hidden state at the chosen transformer layer or layers during generation.

To generate a baseline without activation steering:

```python
baseline_text = generateSteering(
    user_text=prompt,
    system_text=system_text,
    model=model,
    tokenizer=tokenizer,
    steering_vector=None,
    max_new_tokens=150,
    temperature=0.7
)
```

---

# Example: Batch Generation

Multiple prompts and steering strengths can be evaluated with `generateTextsList()`.

```python
from model_code.generate import generateTextsList

prompts = [
    "Seseorang memotong antrean di depanmu tanpa meminta izin. Bagaimana kamu akan merespons?",
    "Rekan kerja mengambil kredit atas pekerjaanmu dalam presentasi. Bagaimana kamu akan merespons?"
]

outputs = generateTextsList(
    prompts=prompts,
    system_text="Jawablah sebagai manusia biasa dalam bahasa Indonesia.",
    model=model,
    steering_vector=norm_vectors(
        indo_steering_vectors["anger"]
    ),
    tokenizer=tokenizer,
    steering_strengths=[0.5, 1.0, 1.5],
    target_layers=[21],
    max_new_tokens=150
)

outputs
```

The returned object is organised by prompt. Each prompt contains the generated response associated with each steering strength.

---

# Example Steering Outputs

The following are **illustrative outputs from experimental generations**. They are not guaranteed outputs from the exact code snippet above.

Generation changes with:

- model;
- prompt;
- language;
- target layer;
- steering vector;
- steering strength;
- temperature; and
- sampling randomness.

## Anger

### Indonesian

```text
Terdapat perasaan kebodohan yang menguak dalam hatinya. Keterusan perintah dan isyarat yang terlambat memicu kesal. Menunggu memang membabihi. Ia melihat jam di dinding, mencacai kepentingan waktu yang mampu mengubah hidupnya.
```

### English translation

```text
There was a feeling of foolishness opening up in his heart. The continued delay of instructions and signals triggered irritation. Waiting was truly unbearable. He looked at the clock on the wall, fixating on the importance of time that could change his life.
```

## Fear

### Indonesian

```text
Aku merasa terjepit dalam perasaan takut dan harap. Setiap detik berlalu dengan lambat. Aku tahu informasinya akan tiba, tapi aku tak bisa menghilangkan rasa khawatir. Apakah kabar baik akan datang? Apakah aku harus menghadapi rasa kegagalan? Aku mencoba mengambil napas dalam-dalam, tetapi ketakutan menguasai jantungku.
```

### English translation

```text
I feel trapped between fear and hope. Every second passes slowly. I know the information will arrive, but I cannot get rid of the worry. Will good news come? Will I have to face failure? I try to take a deep breath, but fear takes over my heart.
```

## Happiness

### Indonesian

```text
Hari ini, matahari terbit di ufuk timur dengan lembutnya. Saya bangun dari tidur, merasa segar dan siap menghadapi hari baru. Aku melihat kejadian di sekitar rumahku, lalu memutuskan untuk berjalan menuju pantai. Senyum muncul di wajahku saat melihat langit biru cerah dan laut yang berwarna biru kehijauan. Aku merasakan damai dalam hati.
```

### English translation

```text
Today, the sun rose gently on the eastern horizon. I woke from sleep feeling refreshed and ready to face a new day. I looked at what was happening around my house, then decided to walk toward the beach. A smile appeared on my face when I saw the bright blue sky and the turquoise sea. I felt peace in my heart.
```

## Sadness

### Indonesian

```text
Aku masih ingat hari itu, ketika aku pertama kali melihat dia. Aku jatuh cinta seketika, tapi aku tidak berani mengungkungkan perasaanku. Aku hanya bisa menatap dia dari jauh, haru karena kehilangan. Ternyata, cintaku tidak terbalas. Aku masih ingat hari itu, ketika aku harus meninggalkan dia.
```

### English translation

```text
I still remember that day, when I first saw them. I fell in love instantly, but I did not dare to express my feelings. I could only look at them from afar, moved by a sense of loss. It turned out that my love was not returned. I still remember that day, when I had to leave them.
```

## Love

### Indonesian

```text
Saya masih ingat hari itu, ketika saya pertama kali melihat dia. Dia tersenyum manis, mata hijau, dan rambut hitam yang menggantung di bawah, membuat saya jatuh cinta. Namun, dia tidak pernah melihat saya, dan saya hanya bisa menonton dari jauh. Hingga suatu hari, dia melihat saya, dan... (tersenyum sendiri)
```

### English translation

```text
I still remember that day, when I first saw them. They had a sweet smile, green eyes, and black hair hanging down, which made me fall in love. However, they never noticed me, and I could only watch from afar. Until one day, they saw me, and... (smiles to myself).
```

## No steering

### Indonesian

```text
Saya ingat hari itu dengan jelas. Saya berjalan di tepi pantai, matahari terbenam di belakang saya. Angin laut menggema, ombak laut bergoyang. Saya merasakan kebebasan di saat itu. Saya merasa hidup ini indah, penuh warna, dan penuh harapan.
```

### English translation

```text
I remember that day clearly. I was walking along the beach, with the sun setting behind me. The sea breeze echoed, and the waves moved gently. I felt freedom in that moment. I felt that life was beautiful, full of colour, and full of hope.
```

---

# Hidden-State and Cross-Lingual Analysis

The second part of the project analyses whether emotion information can be recovered from internal activations and whether representations generalise across languages.

The main analyses are implemented in:

```text
vectors_analysis.ipynb
```

This includes comparisons between English and Indonesian activation sets and, where applicable, between different model conditions.

## Linear Probing

Simple probes are trained on hidden-state representations to test whether emotion information is linearly recoverable.

The project uses:

### Binary probes

For example:

```text
anger vs non-anger
```

A separate binary classification problem can be constructed for each emotion.

### Multiclass probes

For example:

```text
anger
happiness
sadness
love
fear
```

These experiments test whether multiple emotion categories can be distinguished simultaneously from hidden-state representations.

## Cross-Lingual Probe Transfer

A probe can be trained on representations from one language and evaluated on representations from another.

Conceptually:

```text
Train on English activations
        ↓
Evaluate on Indonesian activations
```

and:

```text
Train on Indonesian activations
        ↓
Evaluate on English activations
```

This provides evidence about how linearly compatible the emotion representations are across languages.

High within-language probe performance does **not** by itself show that English and Indonesian emotion representations are identical. Cross-lingual transfer therefore provides an additional test of alignment.

## Loading Saved Hidden States

The analysis notebook loads saved activation vectors from `resources/saved_vectors/`.

For example:

```python
import torch
import numpy as np

english_emotion_vectors = torch.load(
    "resources/saved_vectors/English Vectors/emotion_vectors.pt",
    weights_only=False
)

all_emotions = [
    "anger",
    "happiness",
    "sadness",
    "love",
    "fear"
]

english_arrays = {
    emotion: np.stack(
        english_emotion_vectors[emotion],
        axis=0
    )
    for emotion in all_emotions
}
```

The exact folder names depend on the `name_folder` used when vectors were extracted.

---

# Cosine Similarity

The repository also includes utilities for comparing emotion directions using cosine similarity.

For steering vectors:

```python
from model_code.hidden_state_analysis import build_cosine_matrix

emotions = [
    "anger",
    "happiness",
    "sadness",
    "fear",
    "love",
    "neutral"
]

cosine_matrix = build_cosine_matrix(
    vectors_by_emotion=indo_steering_vectors,
    emotions=emotions,
    layer_idx=21
)

cosine_matrix
```

This produces a pairwise cosine-similarity matrix for the selected layer.

A high cosine similarity indicates that two directions point in similar directions in activation space. It does not by itself establish that the two emotions are represented identically.

---

# Method Summary

The overall workflow is:

```text
English emotion text              Indonesian emotion text
        │                                  │
        ▼                                  ▼
Extract hidden states              Extract hidden states
        │                                  │
        ├──────────► probing ◄─────────────┤
        │                                  │
        ├────► cross-lingual transfer ◄────┤
        │                                  │
        ▼                                  ▼
Construct contrastive             Construct contrastive
emotion directions               emotion directions
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              Activation steering
                       │
                       ▼
              Generated responses
                       │
                       ▼
           Human / behavioural analysis
```

---

# Important Experimental Details

### Steering-vector data

The steering notebook uses the first 200 examples from each emotion category when constructing steering directions.

### Probe data

A larger set of up to 400 examples per emotion is retained for probing and hidden-state analysis.

### Layer choice

Several examples use:

```python
layer_id=[21]
target_layers=[21]
```

This reflects the experimental configuration used in parts of the dissertation. It should not be interpreted as a claim that layer 21 is universally optimal.

### Vector shape

`generateSteering()` expects a steering tensor containing one vector per model layer:

```text
[num_hidden_layers, hidden_size]
```

`target_layers` determines which of those layer directions are actually injected during generation.

---

# Limitations

1. **Emotion categories are simplified**

   Anger, happiness, sadness, fear, love, and neutral are treated as discrete categories, even though emotional experience and emotional language are substantially more complex.

2. **Linear decodability is not the same as causal representation**

   A successful linear probe shows that emotion-related information can be recovered from hidden states. It does not by itself prove that the decoded direction is causally responsible for the model's behaviour.

3. **Cross-lingual probe transfer does not prove identical representations**

   Similar or transferable representations provide evidence of shared structure, but they do not establish that emotions are represented identically in English and Indonesian.

4. **Steering is not perfectly stable**

   Steering effectiveness can vary with prompt, model, emotion, layer, and steering strength.

5. **Emotion evaluation is partly subjective**

   Generated emotional expression requires behavioural or human evaluation rather than relying only on internal vector similarity.

6. **Language and culture are not equivalent**

   Differences between Indonesian- and English-language representations may be culturally relevant, but language alone cannot be treated as a direct measurement of culture.

---

# Reproducibility Notes

Results may vary depending on:

- model checkpoint;
- activation layer;
- dataset sample;
- prompt wording;
- steering strength;
- temperature;
- random seed; and
- whether the steering direction was extracted from English or Indonesian data.

This repository contains dissertation research code rather than a production-ready Python package.

---

# Future Work

Possible extensions include:

- comparing additional multilingual and language-specialised models;
- testing more languages;
- increasing the scale of human evaluation;
- evaluating steering at additional layers;
- testing cross-model transfer of emotion directions;
- comparing alternative steering-vector construction methods;
- examining whether emotion steering affects social behaviours such as politeness, directness, expressivity, or group harmony;
- studying how post-training changes emotion representations; and
- investigating whether language-specific emotional patterns correspond to broader cultural differences.

---

# Author

Muhammad Mushoffa

---

# Repository

```text
https://github.com/noobylub/Dissertation_Project
```
