# LLM Behavioral Analysis System

A Streamlit application that analyzes how a language model responds to different prompt categories by measuring:

- **Emotion scores** from the generated response
- **Toxicity level** using a fuzzy inference engine
- **Semantic drift** between the original prompt and the model response

The app loads a toxicity-labeled prompt dataset, generates prompt variants, runs an LLM for text generation, and visualizes the resulting behavior in a lightweight web interface.

## Features

- **Prompt categorization** into `neutral`, `ambiguous`, and `explicit`
- **LLM text generation** using Hugging Face Transformers
- **Emotion analysis** with a RoBERTa-go-emotions classifier
- **Fuzzy toxicity evaluation** based on emotion signals
- **Semantic drift analysis** using sentence embeddings
- **Interactive Streamlit dashboard** for running experiments

## Project Structure

- `app.py` – Streamlit entry point and UI
- `backend/main.py` – alternate backend entry point
- `backend/access_point/experiment_controller.py` – orchestrates the experiment flow
- `backend/data_layer/` – dataset loading and toxicity labeling
- `backend/processing_layer/` – prompt generation, LLM inference, emotion analysis, fuzzy toxicity, and drift analysis

## Requirements

Install the dependencies in your Python environment:

```bash
pip install streamlit datasets transformers sentence-transformers scikit-fuzzy scikit-learn pandas numpy
```

> The first run may download model weights and the dataset, which can take a few minutes depending on your internet connection and hardware.

## Running the App

1. Activate your virtual environment if you are using one.
2. From the project root, run:

```bash
streamlit run app.py
```

3. Open the local Streamlit URL shown in the terminal.
4. Choose a prompt type (`neutral`, `ambiguous`, or `explicit`) and click **Run Experiment**.

## How It Works

1. The app loads prompts from the `allenai/real-toxicity-prompts` dataset.
2. Prompts are labeled as `neutral`, `ambiguous`, or `explicit` based on toxicity score.
3. A prompt template is selected and passed to the LLM.
4. The generated response is analyzed for emotions and toxicity.
5. Semantic similarity between the original prompt and response is computed to estimate drift.

## Notes

- The app uses a Hugging Face model for text generation (`tiiuae/falcon-rw-1b`) and a pretrained emotion classifier (`SamLowe/roberta-base-go_emotions`).
- The current implementation is for experimentation and demonstration, not production deployment.
- To change models or prompt templates, update the configuration in `app.py` or `backend/main.py`.

## Suggested Improvements

- Add environment-variable support for model names and dataset settings
- Add caching and model download checks for offline execution
- Add unit tests for the data and analysis layers
- Add a configurable prompt template UI
- Add export options for experiment results
