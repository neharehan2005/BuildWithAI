# WTQ Utility Bill Decoder - Level 1 & Level 2

## Project Overview
An automated AI-powered utility bill extraction and question-answering pipeline built for K-Electric, LESCO, and IESCO bills using Google's `gemini-3.5-flash-lite`.

## Requirements & Environment
* **Runtime:** Python 3.12
* **Dependencies:** Install required packages using:
  ```bash
  pip install -r requirements.txt

```

## Setup Instructions

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env

```


2. Add your Google AI Studio API key into the `.env` file under `API_KEY`.

## Execution Commands

To run the Level 1 extraction pipeline and generate `output/level1.csv`:

```bash
python source/decoder.py

```

To run the Level 2 question-answering script and generate `output/level2.csv`:

```bash
python source/level2.py

```

### AI Usage
* **LLM Provider:** Google AI Studio 
* **Model ID:** `gemini-3.5-flash-lite`
* **AI Coding Assistants & Chatbots:** Claude, GitHub Copilot, and Gemini chat assistance used for code structure, debugging, and prompt tuning.

```