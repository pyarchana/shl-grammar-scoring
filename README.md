# SHL Grammar Scoring Engine

Predicts the grammar score (0 to 5) of a 45 to 60 second spoken English clip, for the SHL Research Engineer hiring challenge on Kaggle.

## Approach

1. **Transcribe** every clip with Whisper `medium.en` on a Kaggle GPU, and record simple timing stats (duration, speaking time, long pauses).
2. **Describe** each clip with three groups of inputs:
   - 15 hand-crafted features tied to the rubric: length and speech rate, sentence length, subordinate clauses, vocabulary range, fillers and repetitions, LanguageTool error rates, and sentence acceptability from a RoBERTa model fine-tuned on CoLA
   - pretrained text embeddings of the transcript (MPNet, DeBERTa-v3-large)
   - a pretrained audio embedding (WavLM), averaged over time
3. **Fit** a ridge regression on standardised features, with the penalty strength picked by leave-one-out CV, and clip predictions to 0 to 5.
4. **Choose** which feature groups to keep with 5-fold stratified cross-validation. A larger feature set has to beat a smaller one by more than 0.005 RMSE to be kept.

Heavy models run once on Kaggle and their outputs are cached in `outputs/`, so everything else runs on a laptop CPU in about a minute.

## Results

Filled in after the first full run: 5-fold CV RMSE and Pearson correlation, training RMSE, and the public leaderboard score.

## Repository layout

```
kaggle_transcribe.ipynb  # Kaggle GPU: transcripts, embeddings, acceptability scores -> outputs/
notebook.ipynb           # main notebook: features, CV, training RMSE, plots, report, submission
src/
  data.py                # loading the CSVs and Kaggle outputs, alignment checks, submission writer
  features.py            # hand-crafted transcript, timing and LanguageTool features
  train.py               # ridge model, cross-validation, metrics
outputs/                 # cached transcripts, features, embeddings, submission.csv
data/                    # competition CSVs (not committed)
```

## Reproducing

1. **Kaggle (GPU, about 40 minutes).** Create a notebook from the competition page, import `kaggle_transcribe.ipynb`, set the accelerator to GPU T4 and turn internet on, then *Save & Run All*. Download `shl_outputs.zip` from the output tab and unzip it in the repo root. It contains `outputs/` and a copy of the competition CSVs in `data/`.
2. **Local (CPU).**
   ```bash
   pip install -r requirements.txt
   jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
   ```
   LanguageTool needs Java 17 or newer the first time; after that its results are cached in `outputs/languagetool_features.csv`. The notebook writes `outputs/submission.csv`.

All random seeds are fixed (42), so the numbers in the notebook are reproducible.
