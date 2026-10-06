# SHL Grammar Scoring

Grammar score prediction (0-5) for spoken English audio clips, built for the SHL Research Engineer hiring challenge on Kaggle.

## Approach

- Transcription with Whisper (`medium.en`)
- Features:
  - hand-crafted: word count, speech rate, pauses, sentence length, subordinate clauses, lexical diversity, fillers, repetitions, LanguageTool error rates, CoLA acceptability scores
  - text embeddings: MPNet, DeBERTa-v3-large
  - audio embeddings: WavLM
- Ridge regression with 5-fold stratified cross-validation
- Feature groups selected by CV RMSE

## Structure

```
kaggle_transcribe.ipynb   transcription and embeddings (Kaggle GPU)
notebook.ipynb            features, training, evaluation, report
src/data.py               data loading, submission file
src/features.py           hand-crafted features
src/train.py              model and cross-validation
outputs/                  transcripts, features, embeddings, submission
```

## Running

1. Run `kaggle_transcribe.ipynb` on Kaggle (GPU T4, internet on). Download `shl_outputs.zip` from the output tab and unzip it in the repo root.
2. Install dependencies and run `notebook.ipynb`:

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

LanguageTool requires Java 17+.
