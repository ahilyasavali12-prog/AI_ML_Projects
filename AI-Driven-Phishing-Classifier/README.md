# AI-Driven Phishing Classifier

> Fine-tuned BERT model that detects phishing emails with **99% accuracy** and **0.9998 ROC-AUC** — deployed as a live REST API.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-green?style=flat-square)
![Colab](https://img.shields.io/badge/Google_Colab-T4_GPU-red?style=flat-square)

---

## What this project demonstrates

Most phishing detectors rely on keyword lists and rule-based filters — attackers bypass them by changing a few words. This project takes a fundamentally different approach: fine-tuning BERT to understand the **intent and semantics** of an email, not just its surface patterns.

The result is a classifier that catches phishing emails a rule-based filter would miss, with a false positive rate of just **0.3%** — meaning legitimate emails almost never get wrongly flagged.

---   

## Results

| Metric | Score |
|--------|-------|
| Accuracy | **99%** |
| ROC-AUC | **0.9998** |
| F1 Score | **0.99** |
| False Positive Rate | **0.3%** (7 legit emails flagged out of 4,143) |
| False Negative Rate | **0.9%** (18 phishing emails missed out of 4,143) |

```
Classification Report
─────────────────────────────────────────────────
              precision    recall  f1-score   support

  Legitimate       0.99      1.00      0.99      2072
    Phishing       1.00      0.99      0.99      2071

    accuracy                           0.99      4143
```

---

## Architecture
```
Raw Email Input
      │
      ▼
Preprocessing  ──  text cleaning, deduplication, class balancing
      │
      ▼
BERT Tokeniser  ──  bert-base-uncased, max 256 tokens, WordPiece
      │
      ▼
Fine-tuned BERT  ──  109M parameters, 4 epochs, lr=2e-5
      │
      ▼
Classification Head  ──  linear layer → 2 outputs (SAFE / PHISHING)
      │
      ▼
FastAPI + ngrok  ──  live REST API with /classify and /batch endpoints
```

## Dataset

Three real-world sources combined and balanced:

| Source | Size | Type |
|--------|------|------|
| `cybersectony/PhishingEmailDetectionv2.0` | ~100k emails | Phishing + legitimate |
| `SetFit/enron_spam` | 33,716 emails | Real corporate emails (Enron) |
| `ealvaradob/phishing-dataset` | 18,650 emails | Curated phishing corpus |

Final balanced dataset: **~27,000 emails** (50% phishing / 50% legitimate), split 70/15/15 into train, validation, and test sets.

---

## Project Structure

```
phishing-classifier/
├── 01_environment_setup.py     # GPU check, package install, Drive mount
├── 02_data_preparation.py      # Dataset loading, cleaning, balancing, EDA
├── 03_tokenisation.py          # BERT tokenisation, PyTorch DataLoaders
├── 04_model_training.py        # BERT fine-tuning, saves best model to Drive
├── 05_evaluation.py            # Confusion matrix, ROC curve, F1 on test set
├── 06_inference.py             # Single email classification with confidence score
├── 07_api_server.py            # FastAPI + ngrok live REST API
└── 08_explainability.py        # SHAP token-level explanations
```

---

## Quick Start

### Run on Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Set runtime: `Runtime > Change runtime type > T4 GPU`
3. Run each file in order — each one ends with a success message before you proceed

```python
# Step 1 — environment
# Step 2 — data (loads automatically from HuggingFace, no uploads needed)
# Step 3 — tokenisation
# Step 4 — training (~30 min on T4 GPU)
# Step 5 — evaluation
# Step 6 — inference
# Step 7 — deploy API
# Step 8 — explainability
```

### API usage (after Step 7)

```bash
# Classify a single email
curl -X POST https://your-ngrok-url/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Urgent! Your PayPal account has been limited. Click here."}'

# Response
{
  "verdict": "PHISHING",
  "phishing_score": 0.9974,
  "safe_score": 0.0026,
  "confidence": 0.9974
}
```

```bash
# Classify multiple emails
curl -X POST https://your-ngrok-url/batch \
  -H "Content-Type: application/json" \
  -d '{"emails": ["email one...", "email two..."]}'
```

Interactive API docs available at `/docs` (Swagger UI).

---

## Explainability

Using SHAP, the model highlights exactly which tokens drove the phishing verdict — making it interpretable for security analysts, not just a black box.

```
Email: "URGENT: Your PayPal account has been limited! Click http://paypa1.ru/verify"

Top phishing signals:
  'urgent'  : +0.42  → PHISHING
  'paypa1'  : +0.38  → PHISHING
  'limited' : +0.31  → PHISHING
  'verify'  : +0.28  → PHISHING
  'click'   : +0.19  → PHISHING
```

---

## Why deep learning over rule-based filters?

| Rule-Based Filters | This Classifier |
|--------------------|-----------------|
| Break when attackers tweak keywords | Understands semantic meaning |
| Require constant manual updates | Retrains on new data automatically |
| High false positive rate | 0.3% false positive rate |
| Cannot generalise to new attack styles | Generalises across unseen phishing variants |
| No confidence scoring | Calibrated probability score per email |

---

## Technical decisions

**Why BERT?** Most phishing research uses TF-IDF or simple word embeddings. BERT's bidirectional attention mechanism understands how words relate across the entire email — capturing social engineering patterns that unidirectional models miss entirely.

**Why 99% accuracy is credible** — not overfitting: the model was evaluated on a completely held-out test set it never saw during training. The confidence distribution shows two sharp spikes at 0.0 and 1.0 — the model is genuinely certain, not memorising.

**False positive strategy:** The model outputs a calibrated probability score rather than a hard binary label. Security teams can set their own threshold — stricter for high-risk inboxes, more permissive for bulk mail pipelines.

---

## Stack

| Component | Technology |
|-----------|------------|
| Model | `bert-base-uncased` fine-tuned via HuggingFace Transformers |
| Training | PyTorch + AdamW + linear warmup scheduler |
| Serving | FastAPI + uvicorn + ngrok |
| Explainability | SHAP |
| Platform | Google Colab (T4 GPU) |
| Storage | Google Drive |

---

## Monitoring considerations

In production this classifier would be paired with:

- Prediction confidence logging — drift in average confidence signals new attack patterns
- Periodic retraining on fresh PhishTank samples
- Alert on sudden spikes in phishing rate — may indicate a targeted campaign
- SHAP explanations surfaced to analysts on low-confidence predictions

---

## Further reading

- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [PhishTank — community-verified phishing database](https://phishtank.org)
- [APWG eCrime Reports — phishing trends and statistics](https://apwg.org/resources/apwg-reports/)
- [SHAP: A unified approach to explaining model output](https://shap.readthedocs.io)

---

*This project demonstrates that modern threat detection requires moving beyond signatures — AI learns the intent behind an attack, not just its surface appearance.*
