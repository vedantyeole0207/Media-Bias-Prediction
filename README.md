# Political Media Bias — Project Report

## Overview

This repository/notebook tackles **binary classification of political social-media posts** (or short articles) to detect media bias. The work explores both traditional feature-based models (TF–IDF + Logistic Regression and other classical models) and a modern transformer-based approach (BERT via `keras_nlp`). The notebook's conclusion: **BERT outperforms TF–IDF-based models** on this dataset.

---

## Problem statement

Detect whether a piece of political text (social media post or short article) is **neutral** or **partisan**. This is framed as a binary classification problem:

* `neutral` → label `0`
* `partisan` → label `1`

The goal is to compare classical NLP pipelines (TF–IDF + classical ML) with a transformer-based classifier (BERT) and evaluate which approach yields better performance on held-out data.

---

## Dataset

* **Filename used in notebook:** `political_social_media.csv`
* **Key columns used:**

  * `text` — the raw text to classify
  * `bias` — the string label (`'neutral'` or `'partisan'`)

The notebook keeps only these two columns and drops missing values:

```python
data = data[['text', 'bias']].dropna()
```

The label mapping used is:

```python
data['label'] = data['bias'].map({'neutral': 0, 'partisan': 1})
```

> The notebook inspects class balance using `data['bias'].value_counts()` and trains with a stratified split to preserve label proportions.

---

## Key preprocessing steps

Preprocessing in the notebook is concise and focused on removing noisy tokens (URLs, mentions, hashtags):

```python
import re

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", '', text)
    return text.strip()

data['clean_text'] = data['text'].apply(clean_text)
```

Notes:

* The notebook uses `clean_text` to remove URLs and simple social media artifacts.
* It also sets aside `clean_text` for feature extraction. It does not perform (in the visible cells) advanced normalization such as:

  * tokenization beyond vectorizers
  * explicit lowercasing (TF–IDF does this internally)
  * lemmatization/stemming in the pipeline (though `nltk` is imported and a lemmatizer is present in imports)
  * explicit stopword removal beyond TF–IDF's `stop_words='english'` option

---

## Experimental setup

### Train / test split

The code uses scikit-learn's `train_test_split`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    data['clean_text'], data['label'], test_size=0.2, random_state=42, stratify=data['bias'])
```

* Test size: 20%
* Stratification: by `bias` column
* Random state: 42 (reproducible)

### TF–IDF + classical models

* **Vectorizer:** `TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')`
* Example model used in the notebook: `LogisticRegression(max_iter=300)` (trained on TF–IDF features)
* Evaluation: `accuracy_score` and `classification_report` (precision/recall/f1)

### BERT (transformer) approach

* The notebook loads a BERT-based classifier via `keras_nlp`:

```python
model2 = keras_nlp.models.BertClassifier.from_preset("bert_base_en_uncased", num_classes=2)
```

* Data is converted into `tf.data.Dataset` objects and batched (batch size 16):

```python
train_ds = tf.data.Dataset.from_tensor_slices((train_texts, train_labels)).shuffle(1000).batch(16)
test_ds = tf.data.Dataset.from_tensor_slices((test_texts, test_labels)).batch(16)
```

* Training run in the notebook uses `epochs=3`:

```python
history = model2.fit(train_ds, validation_data=test_ds, epochs=3)
```

* Final predictions use `tf.argmax` on model logits.

---

## Metrics and evaluation

The notebook primarily uses:

* `accuracy_score`
* `classification_report` (precision, recall, f1 for each class)

Printed outputs in the notebook indicate accuracy for the TF–IDF + Logistic Regression run and for the BERT model, and the notebook concludes that **BERT gives superior performance**. (Exact numeric scores were not programmatically extracted from the static notebook during this report generation — please run the notebook to see the printed numeric metrics.)

---

## Notable code / implementation details

* `political_social_media.csv` is loaded via a `file_path` variable.
* Cleaning is intentionally simple (mainly URL, mention, hashtag removal).
* TF–IDF uses unigrams + bigrams, capped at 5,000 features.
* The notebook shows an end-to-end comparison: preprocessing → TF–IDF → Logistic Regression → evaluate, then transformer-based training and evaluation.

---

## How to reproduce (suggested README instructions)

1. **Create a virtual environment** (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. **Install dependencies** (example `requirements.txt`):

```
pandas
scikit-learn
nltk
tensorflow
keras-nlp
lightgbm
matplotlib
jupyter
```

(Version pinning recommended for TF/Keras compatibility.)

3. **Download / place the dataset**

* Put `political_social_media.csv` in the notebook folder (or update `file_path` in the notebook).

4. **Run the notebook**

```bash
jupyter lab  # or jupyter notebook
```

Open `Political_Media_Bias.ipynb` and run the cells in order.

---

## Suggested improvements / next steps

1. **Stronger preprocessing**

   * Expand cleaning: lowercasing, punctuation removal, more robust tokenization.
   * Use lemmatization or stemming and experiment with removing/keeping stopwords.
2. **Data balancing**

   * If classes are imbalanced, try SMOTE, class-weighted loss, or focal loss for BERT.
3. **More model baselines**

   * Add Naive Bayes, Random Forest, and LightGBM as additional baselines (LightGBM is already imported in the notebook).
4. **Hyperparameter tuning**

   * GridSearch or RandomizedSearch for TF–IDF `max_features`, `ngram_range`, and Logistic Regression `C`.
   * For BERT, tune learning rate, batch size, and number of epochs; use a validation split and early stopping.
5. **Error analysis**

   * Save misclassified examples for qualitative analysis.
   * Examine confusion matrix and class-wise performance.
6. **Model serving**

   * Add an inference script or a small FastAPI app to serve the best model for predictions.
7. **Data provenance & license**

   * Document source of `political_social_media.csv` and any licensing constraints.

---

## Files you should include in the GitHub repo

* `Political_Media_Bias.ipynb` (the notebook itself)
* `political_social_media.csv` (if allowed; otherwise provide instructions to download or a small sample)
* `README.md` (a shorter version of this report)
* `requirements.txt`
* `inference.py` or `app.py` (optional: scripts for batch inference or serving)

---

## Final notes

* The notebook provides a clean, readable comparison between TF–IDF-based classical pipelines and a transformer approach. The conclusion that **BERT works better** is consistent with expectations for tasks requiring semantic/contextual understanding.
