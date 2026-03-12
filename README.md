# Cross-Lingual Vectors Dissertation 

## Environment Setup

### 1. Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables Setup
1. Copy the `.env` file and update it with your actual Hugging Face token:
```bash
cp .env .env.local
```

2. Get your Hugging Face token from: https://huggingface.co/settings/tokens

3. Edit `.env.local` and replace `your_huggingface_token_here` with your actual token

4. Load environment variables in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. Hugging Face Authentication
```python
from huggingface_hub import login
import os

# Method 1: Using environment variable
login(token=os.getenv("HF_TOKEN"))

# Method 2: Direct login (will prompt for token)
# login()
```

## Usage

### Working with Gated Models
For accessing gated models like Llama:
1. Request access on Hugging Face
2. Set up your HF_TOKEN in `.env.local`
3. Use the authentication code in your notebook

### Cache Management
Models are cached to avoid re-downloading:
- Default cache location: `./cache/`
- Can be customized via environment variables

## Deactivation
```bash
deactivate
```
