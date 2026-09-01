# Emotions Across Languages in Large Language Models

Code and experimental resources for the MSc dissertation:

**Emotions Across Languages in Large Language Models:  
A Cross-Lingual Emotional Representation Study on Large Language Models**

University of Manchester, 2026.

This repository investigates how emotion-related information is represented
inside multilingual large language models and whether emotion representations
derived from English and Indonesian text can be used for activation steering.

The project combines:

- layer-wise linear probing;
- cross-lingual probe transfer;
- emotion-vector extraction;
- cosine-similarity analysis;
- activation steering; and
- Indonesian text generation for human evaluation.

The five target emotions are:

- anger
- fear
- happiness
- love
- sadness

A sixth **neutral** category is included during probing and contrastive
steering-vector construction.

---

## Research Questions

The experiments address three main questions:

1. To what extent is emotion-related information linearly represented in
   English and Indonesian hidden states?

2. To what extent is the internal representation of emotion shared between
   English and Indonesian?

3. Can emotion vectors derived from English and Indonesian text be used to
   steer model outputs towards the intended emotion?

---

# Repository Structure

```text
Dissertation_Project/
│
├── activation_steering.ipynb
│   Main notebook for loading the models, extracting hidden states and
│   steering vectors, and generating steered responses.
│
├── hidden_state_analysis.ipynb
│   Main notebook for layer-wise probing, cross-lingual probe transfer,
│   and emotion-vector similarity analysis.
│
├── model_code/
│   ├── __init__.py
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
│   Generated model responses used in the steering experiment.
│
├── images/
│   Figures and visualisations produced during analysis.
│
├── no_need/
│   Older/exploratory notebooks retained for reference. These are not
│   required for reproducing the main experiments.
│
├── requirements.txt
└── README.md
