# Hindi–Bhili machine translation experiment

An experimental notebook for fine-tuning `facebook/nllb-200-distilled-600M` on Hindi-to-Bhili sentence pairs. The workflow loads a parallel CSV dataset, creates train/validation/test splits, fine-tunes a sequence-to-sequence model, and evaluates generated translations with corpus BLEU and chrF.

## What is in this repository

- [`Model_try.ipynb`](Model_try.ipynb): data preparation, tokenization, training, inference, and evaluation in Google Colab.
- No training dataset, model weights, or saved evaluation report are included. The notebook references a local Colab CSV path, so it needs your own appropriately licensed data before it can run.

## Data and setup

The notebook expects a CSV with `Hindi` and `Bhili` columns. Update the `pd.read_csv(...)` path to point to your copy of the dataset. It uses Python, PyTorch, Hugging Face Transformers and Datasets, SentencePiece, Accelerate, pandas, and SacreBLEU. A GPU-backed Colab runtime is recommended for this model.

## Workflow

1. Load the sentence pairs and remove rows with missing text.
2. Split the data into training, validation, and test sets.
3. Tokenize Hindi inputs and Bhili targets, then fine-tune NLLB-200 with `Seq2SeqTrainer`.
4. Generate translations on the test set and calculate BLEU and chrF.

## Limitations

This is a research experiment, not a hosted translation service. Bhili is not assigned a dedicated NLLB language code in the notebook; it uses `hin_Deva` for both source and target tokenization. Reported quality should therefore be evaluated on held-out Bhili data before any production use. The notebook's saved model and reports are local outputs and are not part of this repository.
