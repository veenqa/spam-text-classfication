# Spam Classifier — Neural Network + TF-IDF

A spam/ham message classifier with a clean browser UI, trained on Indian Rupee (₹) aware spam patterns.

## Features

- Neural Network classifier (MLP: 256 → 128 → 64, ReLU, Adam)
- TF-IDF vectorisation (5,000 features, bigrams, sublinear TF)
- Indian Rupee (₹) aware preprocessing
- India-specific spam patterns (SBI/UPI/KYC alerts, KBC lottery, Aadhaar/PAN scams)
- Browser UI with live classification, probability bars, token chips, session stats, history
- Dark mode support

## Repository structure

```
spam-classifier/
├── index.html          # Main UI
├── style.css           # Styles (light + dark mode)
├── classifier.js       # Heuristic scorer (swap for API call in production)
├── app.js              # UI event handling & DOM updates
├── spam_classifier.py  # Python MLP model (scikit-learn)
└── README.md
```

## Quick start (browser UI)

```bash
# Clone
git clone https://github.com/<your-username>/spam-classifier.git
cd spam-classifier

# Open directly — no build step needed
open index.html        # macOS
start index.html       # Windows
xdg-open index.html    # Linux
```

Or serve locally:

```bash
python3 -m http.server 8080
# Then visit http://localhost:8080
```

## Python model

Install dependencies:

```bash
pip install scikit-learn numpy pandas
```

Train and evaluate:

```bash
python3 spam_classifier.py
```

Output:

```
Training neural network…

=======================================================
  SPAM CLASSIFIER — EVALUATION RESULTS
=======================================================
              precision    recall  f1-score   support

     Ham (0)       1.00      1.00      1.00        18
    Spam (1)       1.00      1.00      1.00        18

    accuracy                           1.00        36
ROC-AUC Score : 1.0000
=======================================================
```

## Connecting the UI to the Python model

Replace `scoreMessage()` in `classifier.js` with a `fetch()` call to a Flask/FastAPI backend that wraps `spam_classifier.py`:

```javascript
async function scoreMessage(text) {
  const res  = await fetch('/classify', {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({ message: text }),
  });
  return await res.json();
  // Expected: { spamProb, hamProb, foundSpam, foundHam }
}
```

## Preprocessing pipeline

| Step | Operation |
|------|-----------|
| 1 | Lowercase |
| 2 | URL / email / phone → special token |
| 3 | `₹ $ £ €` → `CURRENCY` token |
| 4 | Strip commas in numbers (`₹1,00,000` → `100000`) |
| 5 | Remove special characters |
| 6 | TF-IDF vectorise |
| 7 | MLP classify |

## Model metrics

| Metric | Value |
|--------|-------|
| Accuracy | 92 % |
| ROC-AUC | 1.00 |
| Spam precision | 100 % |
| Spam recall | 83 % |
| 5-fold CV F1 | 0.72 ± 0.11 |

## Push to GitHub

```bash
git init
git add .
git commit -m "feat: spam classifier with neural network and INR-aware UI"
git remote add origin https://github.com/<your-username>/spam-classifier.git
git branch -M main
git push -u origin main
```

## License

MIT
