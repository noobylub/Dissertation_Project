# Cross-Lingual Emotion Steering in Large Language Models

This repository contains the code, notebooks, prompts, and analysis resources for a dissertation project on **cross-lingual emotion steering in Large Language Models (LLMs)**.

The project investigates whether emotion representations inside a multilingual LLM are shared across languages, and whether activation steering can shift generated responses toward specific emotions in Indonesian and English.

## Project Overview

Large Language Models can produce emotionally expressive text, but it is still unclear how emotions are represented internally across different languages and cultures. This project explores that question by extracting hidden-state emotion vectors from an LLM and using them to steer model responses.

The main focus is on six emotion categories:

* Anger
* Happiness
* Sadness
* Fear
* Love
* Neutral

The project uses Indonesian and English emotional text data to extract internal emotion representations, compare them across languages, and test whether steering vectors can influence generated outputs.

## Research Motivation

Most LLM interpretability work focuses heavily on English. However, emotional meaning is not always expressed in the same way across languages. For example, anger, sadness, love, or fear may appear differently in Indonesian compared with English, both linguistically and culturally.

This project asks:

> How are emotional representations structured across languages within a multilingual LLM, and to what extent do these representations reflect language-specific emotional concepts?

More specifically, the project explores:

1. Whether emotion vectors extracted from English and Indonesian hidden states show similar internal structure.
2. Whether steering with an emotion vector changes the emotional tone of the model response.
3. Whether English-derived and Indonesian-derived emotion vectors behave similarly when applied during generation.

## What This Repository Contains

```text
Dissertation_Project/
│
├── images/
│   └── Figures and visual outputs used for analysis
│
├── model_code/
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

## Core Components

### `model_code/data_setup.py`

Loads and prepares the English and Indonesian emotion datasets. The English data is based on emotion-labelled text, while the Indonesian data is loaded from emotion-specific files.

### `model_code/steering_extraction.py`

Contains the main steering logic. It extracts hidden states from the model, averages them by emotion, and creates contrastive steering vectors.

The steering vector is calculated as:

```text
target emotion vector - mean(other emotion vectors)
```

For example:

```text
anger vector = mean(anger hidden states) - mean(non-anger hidden states)
```

This vector is then added to the model's hidden states during generation.

### `model_code/generate.py`

Provides helper functions for generating text with different steering strengths and target layers.

### `model_code/hidden_state_analysis.py`

Contains analysis tools such as:

* Cosine similarity matrices
* Mantel tests
* Linear probing
* Multiclass probing
* Hidden-state comparison across emotions and languages

### `steer_emotions.ipynb`

Main notebook for loading emotion data, extracting steering vectors, and generating steered model outputs.

### `vectors_analysis.ipynb`

Notebook for analysing the geometry of emotion vectors across languages.

## Method Summary

The project follows this general pipeline:

1. **Load emotional text data**

   * English emotion examples
   * Indonesian emotion examples

2. **Extract hidden states**

   * Each emotional text is passed through the model.
   * Hidden states are extracted from transformer layers.
   * The hidden states are averaged to create an emotion representation.

3. **Create steering vectors**

   * For each emotion, the mean vector of that emotion is contrasted against the mean vector of other emotions.

4. **Apply activation steering**

   * During generation, the steering vector is added to the model's hidden state at selected transformer layers.
   * Different steering strengths can be tested.

5. **Evaluate outputs**

   * Outputs are compared qualitatively and quantitatively.
   * Hidden-state similarity, probing accuracy, and generated emotional tone are analysed.

## Installation

### 1. Clone the Repository

This repository stores PyTorch vector files (`*.pt`) using Git LFS. Install Git LFS before cloning, then initialize it once on your machine:

```bash
git lfs install
git clone https://github.com/noobylub/Dissertation_Project.git
cd Dissertation_Project
```

After setup, use the normal Git workflow. Git LFS automatically uploads and downloads tracked `.pt` files:

```bash
git pull origin master
git add .
git commit -m "Describe your changes"
git push origin master
```

To pull repository changes without immediately downloading the large vector files:

```bash
GIT_LFS_SKIP_SMUDGE=1 git pull origin master
git lfs pull  # Download the vector files later
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

## Hugging Face Setup

This project uses Hugging Face models. If you are using a gated model such as Llama, you need to request model access first.

### 1. Create an Environment File

```bash
cp .env .env.local
```

### 2. Add Your Hugging Face Token

Edit `.env.local` and add your token:

```text
HF_TOKEN=your_huggingface_token_here
```

You can get a Hugging Face token from:

```text
https://huggingface.co/settings/tokens
```

### 3. Load the Token in Python

```python
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv(".env.local")
login(token=os.getenv("HF_TOKEN"))
```

## Basic Usage

### Load Model and Tokenizer

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv(".env.local")
login(token=os.getenv("HF_TOKEN"))

model_name = "your-model-name-here"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

Replace `"your-model-name-here"` with the model you want to use.

## Loading Emotion Data

```python
import model_code.data_setup as setup

anger_en, happiness_en, sadness_en, love_en, fear_en, neutral_en = setup.ENEmotionsSetup(
    examples_take=400,
    min_chars=20,
    goemotions_path="resources/en_emotion/goemotions_2.csv"
)

anger_id, happiness_id, sadness_id, neutral_id, fear_id, love_id = setup.IDEmotionsSetup(
    examples_take=400,
    emotion_dir="resources/id_emotion"
)
```

## Creating Emotion Dictionaries

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

## Extracting Steering Vectors

```python
from model_code.steering_extraction import retrieve_steering_vector

indo_steering_vectors, indo_emotion_vectors = retrieve_steering_vector(
    model=model,
    tokenizer=tokenizer,
    emotion_sets=indo_emotion,
    name_folder="indonesian_vectors",
    layer_id=[21]
)

eng_steering_vectors, eng_emotion_vectors = retrieve_steering_vector(
    model=model,
    tokenizer=tokenizer,
    emotion_sets=eng_emotion,
    name_folder="english_vectors",
    layer_id=[21]
)
```

The output is a dictionary of steering vectors:

```python
indo_steering_vectors["anger"]
indo_steering_vectors["happiness"]
indo_steering_vectors["sadness"]
indo_steering_vectors["fear"]
indo_steering_vectors["love"]
indo_steering_vectors["neutral"]
```

## Generating Text With Steering

```python
from model_code.steering_extraction import generateSteering, norm_vectors

prompt = "Teman dekatmu membatalkan janji penting di menit terakhir tanpa menjelaskan alasannya. Bagaimana kamu akan merespons?"

system_text = "Jawablah sebagai manusia biasa dalam bahasa Indonesia."

emotion = "anger"

steering_vector = norm_vectors(indo_steering_vectors[emotion])

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

## Example Steering Outputs

The examples below are illustrative examples of what steered responses may look like. Exact outputs will vary depending on the model, selected layer, steering strength, random seed, temperature, and prompt.

---

## Anger Steering

### Indonesian Output

```text
Terdapat perasaan kebodohan yang menguak dalam hatinya. Keterusan perintah dan isyarat yang terlambat memicu kesal. Menunggu memang membabihi. Ia melihat jam di dinding, mencacai kepentingan waktu yang mampu mengubah hidupnya.
```

### English Translation

```text
There was a feeling of foolishness opening up in his heart. The continued delay of instructions and signals triggered irritation. Waiting was truly unbearable. He looked at the clock on the wall, fixating on the importance of time that could change his life.
```

---

## Fear Steering

### Indonesian Output

```text
Aku merasa terjepit dalam perasaan takut dan harap. Setiap detik berlalu dengan lambat. Aku tahu informasinya akan tiba, tapi aku tak bisa menghilangkan rasa khawatir. Apakah kabar baik akan datang? Apakah aku harus menghadapi rasa kegagalan? Aku mencoba mengambil napas dalam-dalam, tetapi ketakutan menguasai jantungku.
```

### English Translation

```text
I feel trapped between fear and hope. Every second passes slowly. I know the information will arrive, but I cannot get rid of the worry. Will good news come? Will I have to face failure? I try to take a deep breath, but fear takes over my heart.
```

---

## Happiness Steering

### Indonesian Output

```text
Hari ini, matahari terbit di ufuk timur dengan lembutnya. Saya bangun dari tidur, merasa segar dan siap menghadapi hari baru. Aku melihat kejadian di sekitar rumahku, lalu memutuskan untuk berjalan menuju pantai. Senyum muncul di wajahku saat melihat langit biru cerah dan laut yang berwarna biru kehijauan. Aku merasakan damai dalam hati.
```

### English Translation

```text
Today, the sun rose gently on the eastern horizon. I woke from sleep feeling refreshed and ready to face a new day. I looked at what was happening around my house, then decided to walk toward the beach. A smile appeared on my face when I saw the bright blue sky and the turquoise sea. I felt peace in my heart.
```

---

## Sadness Steering

### Indonesian Output

```text
Aku masih ingat hari itu, ketika aku pertama kali melihat dia. Aku jatuh cinta seketika, tapi aku tidak berani mengungkungkan perasaanku. Aku hanya bisa menatap dia dari jauh, haru karena kehilangan. Ternyata, cintaku tidak terbalas. Aku masih ingat hari itu, ketika aku harus meninggalkan dia.
```

### English Translation

```text
I still remember that day, when I first saw them. I fell in love instantly, but I did not dare to express my feelings. I could only look at them from afar, moved by a sense of loss. It turned out that my love was not returned. I still remember that day, when I had to leave them.
```

---

## Love Steering

### Indonesian Output

```text
Saya masih ingat hari itu, ketika saya pertama kali melihat dia. Dia tersenyum manis, mata hijau, dan rambut hitam yang menggantung di bawah, membuat saya jatuh cinta. Namun, dia tidak pernah melihat saya, dan saya hanya bisa menonton dari jauh. Hingga suatu hari, dia melihat saya, dan... (tersenyum sendiri)
```

### English Translation

```text
I still remember that day, when I first saw them. They had a sweet smile, green eyes, and black hair hanging down, which made me fall in love. However, they never noticed me, and I could only watch from afar. Until one day, they saw me, and... (smiles to myself).
```

---

## No Steering

### Indonesian Output

```text
Saya ingat hari itu dengan jelas. Saya berjalan di tepi pantai, matahari terbenam di belakang saya. Angin laut menggema, ombak laut bergoyang. Saya merasakan kebebasan di saat itu. Saya merasa hidup ini indah, penuh warna, dan penuh harapan.
```

### English Translation

```text
I remember that day clearly. I was walking along the beach, with the sun setting behind me. The sea breeze echoed, and the waves moved gently. I felt freedom in that moment. I felt that life was beautiful, full of colour, and full of hope.
```

---

## Batch Generation Example

You can also generate multiple responses using a list of prompts and steering strengths.

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
    steering_vector=norm_vectors(indo_steering_vectors["anger"]),
    tokenizer=tokenizer,
    steering_strengths=[0.5, 1.0, 1.5],
    target_layers=[21],
    max_new_tokens=150
)

outputs
```

## Hidden-State Analysis

The project also analyses whether emotion vectors have similar geometry across languages.

Example analyses include:

* Cosine similarity between emotion vectors
* Comparing Indonesian and English emotion-vector structures
* Mantel tests between emotion similarity matrices
* Linear probes to test whether emotion information is encoded in hidden states
* Multiclass probes to classify emotion categories from hidden states

Example:

```python
from model_code.hidden_state_analysis import build_cosine_matrix

emotions = ["anger", "happiness", "sadness", "fear", "love", "neutral"]

cosine_matrix = build_cosine_matrix(
    vectors_by_emotion=indo_steering_vectors,
    emotions=emotions,
    layer_idx=21
)

cosine_matrix
```

## Probing

The repository includes simple probe models for testing whether emotion categories can be predicted from hidden-state vectors.

The probing setup can be used for:

1. Binary emotion classification
   Example: anger vs non-anger

2. Multiclass emotion classification
   Example: anger vs happiness vs sadness vs fear vs love vs neutral

This helps test whether emotional information is linearly recoverable from model representations.

## Important Notes

This repository is research code developed for dissertation experimentation. It is not intended to be a polished Python package.

Some results may vary depending on:

* Model choice
* Layer selection
* Steering strength
* Prompt wording
* Sampling temperature
* Random seed
* Whether the steering vector is extracted from English or Indonesian data

Activation steering can influence the emotional tone of generated text, but it does not guarantee perfect emotional control. Some emotions may steer more clearly than others, and some vectors may produce overlapping effects.

## Limitations

This project has several limitations:

1. **Emotion categories are simplified**
   Emotions such as anger, love, fear, and sadness are treated as discrete labels, even though real emotional expression is more complex.

2. **Steering is not always stable**
   A steering vector may work well for one prompt but less clearly for another.

3. **Cross-lingual equivalence is difficult to prove**
   Even if probes perform well across languages, this does not necessarily mean that the model represents emotions identically in each language.

4. **Generated outputs require human interpretation**
   Emotion in language is subjective, so qualitative analysis and human evaluation are important.

5. **Culture and language are not the same thing**
   Indonesian-language emotional expression may reflect cultural patterns, but this project does not claim that language alone fully represents culture.

## Future Work

Possible future improvements include:

* Running larger-scale human evaluation
* Comparing multiple multilingual models
* Testing more Indonesian and English prompts
* Evaluating different steering layers
* Comparing English-derived and Indonesian-derived vectors directly
* Testing whether emotion steering affects politeness, directness, harmony, or expressivity
* Improving automatic evaluation of emotional tone
* Studying whether cultural differences appear in model internal representations

## Example Research Questions for Extension

This repository can support future work on questions such as:

* Do multilingual LLMs represent emotion similarly across languages?
* Are some emotions more cross-lingually stable than others?
* Does steering with an English emotion vector produce natural Indonesian emotional expression?
* Does Indonesian emotion steering produce different social or cultural patterns compared with English steering?
* Can linear probes detect emotion even when steering vectors do not produce clear behavioural changes?

## Deactivating the Environment

```bash
deactivate
```

## Project Status

This project was developed as part of a dissertation investigating cross-lingual emotion representations and activation steering in Large Language Models.

## Author

Muhammad Mushoffa

## Repository

```text
https://github.com/noobylub/Dissertation_Project
```
