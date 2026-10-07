# SHL Grammar Scoring

Grammar score prediction (0-5) for spoken English audio clips, built for the SHL Research Engineer hiring challenge on Kaggle.

## Approach

- Transcription with Whisper (`medium.en`)
- Features:
  - hand-crafted: word count, speech rate, pauses, sentence length, subordinate clauses, lexical diversity, fillers, repetitions, LanguageTool error rates, CoLA acceptability scores
  - text embeddings: DeBERTa-v3-large (MPNet tested, not used)
  - audio embeddings: WavLM, Whisper large-v3 encoder
- One ridge regression per feature group, blended with non-negative weights
- 5-fold stratified cross-validation for every modelling choice
- The 37 score-0 clips (a separate batch, only in train, absent from test) are excluded

## Results

| | RMSE | Pearson |
|---|---|---|
| 5-fold CV | 0.512 | 0.864 |
| Training set | 0.371 | 0.933 |
| Predict the mean | 1.014 | |
| Public leaderboard (v1) | 0.402 | |
| Public leaderboard (v2) | 0.381 | |

## Structure

```
kaggle_transcribe.ipynb       transcription, text embeddings, CoLA, WavLM (Kaggle GPU)
kaggle_whisper_encoder.ipynb  Whisper large-v3 encoder embeddings (Kaggle GPU)
notebook.ipynb                features, training, evaluation, report
src/data.py                   data loading, submission file
src/features.py               hand-crafted features
src/train.py                  models and cross-validation
outputs/                      transcripts, features, embeddings, submission
```

## Running

1. Run `kaggle_transcribe.ipynb` and `kaggle_whisper_encoder.ipynb` on Kaggle (GPU T4, internet on). Unzip `shl_outputs.zip` in the repo root and put `emb_audio_whisper.npz` in `outputs/`.
2. Install dependencies and run `notebook.ipynb`:

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

LanguageTool requires Java 17+. The cached outputs in `outputs/` are enough to run step 2 without Kaggle.
