# Implementation notes

## Runtime flow

1. `app.py` loads `results.zip` from the repository root once with Streamlit's resource cache.
2. `src/inference.py` reads `absa_best_model.pt` and `absa_tfidf.pkl` from that archive.
3. The saved TF-IDF vectorizer transforms comments into sparse character n-gram features. Inference batches comments, converts each batch to float32 dense arrays, and runs the PyTorch model on CPU.
4. The classifier predicts one of four states for each of the ten aspects: Absent, Negative, Neutral, Positive. The UI hides Absent predictions.
5. `src/dashboard.py` aggregates only predictions currently held in the active Streamlit session.

## Data split and leakage controls

The Kaggle notebook uses the dataset's provided Train, Validation/Dev, and Test splits (7,786 / 1,112 / 2,224 records before exact duplicate cleanup). Ten exact duplicate rows were removed from Train, leaving 7,776. The TF-IDF vectorizer is fitted on Train only and then applied unchanged to validation and test. Validation is used for model selection; the test split is reserved for final evaluation. Do not tune thresholds or model settings against test metrics.

The model is a single linear layer over character TF-IDF features (2–5 character n-grams, minimum document frequency 3, up to 30,000 features, sublinear term frequency). The output is reshaped to ten independent four-state predictions.

## Reported test metrics

The notebook reports Pair F1 0.7007, Macro Pair F1 0.4484, exact match 0.2491, and state accuracy 0.8595. Macro performance is materially lower than aggregate Pair F1; use predictions as review aids and inspect errors for rare aspects.

## Artifact archive

`results.zip` contains:

- `absa_best_model.pt`: PyTorch state dict and checkpoint metadata.
- `absa_tfidf.pkl`: fitted scikit-learn vectorizer.
- `absa_slp_history.csv`: training history.
- Notebook-generated metric plots.

Only load this pickle from a trusted project artifact. Python pickle is not safe for untrusted files. The archive is the deployment artifact; data CSVs are not required for inference and are not included.

## Dashboard scope

The web app does not connect to product-review platforms and does not contain the full 11,122-comment dataset. Metrics describe only the current user-supplied batch. Streamlit session data is transient; the app has no database or persistent review store.

## Known dataset limitation

UIT-ViSFD has no dedicated temperature/overheating aspect. A comment about a hot device may be assigned to performance/general or missed. The model is not a temperature classifier.
