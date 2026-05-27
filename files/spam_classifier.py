"""
Spam Classifier using Neural Network + TF-IDF
============================================================
Architecture:
  Text → TF-IDF Vectorizer → MLP Neural Network → Spam/Ham

Dependencies: scikit-learn, numpy, pandas
Install: pip install scikit-learn numpy pandas
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
import re
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
#  1. DATA
# ─────────────────────────────────────────

SPAM_MESSAGES = [
    "Congratulations! You've won ₹83,00,000! Click here to claim now",
    "FREE MEDICINE! Buy now, massive discounts, limited time offer",
    "You have been selected! Call now to claim your prize",
    "Make money fast! Work from home earn ₹40,000/week guaranteed",
    "URGENT: Your bank account suspended. Click to verify immediately",
    "Hot singles in your area want to meet you tonight",
    "Buy cheap medications online no prescription needed",
    "You won the KBC lottery! Send bank details to claim ₹50,00,000",
    "Lose 10 kg in 30 days! Amazing Ayurvedic weight loss pill",
    "FREE iPhone 15! You are today's lucky winner click here",
    "Investment opportunity! 500% returns guaranteed overnight",
    "Send your Aadhaar and PAN to receive ₹2,00,000 government grant",
    "Earn ₹80,000 per week working from home no experience needed",
    "Your credit card has been charged ₹41,000. Call 1800-SCAM-NOW",
    "Cheap replica watches free shipping all over India",
    "Win a vacation to Goa! You have been selected as winner",
    "Double your investment guaranteed returns crypto scheme",
    "Amazing new herbal pill doctors hate this one weird trick",
    "Free gift card ₹2,000 Flipkart just complete this survey",
    "Get rich quick with our revolutionary trading system",
    "Herbal supplements cure diabetes heart disease guaranteed",
    "You have unclaimed LIC insurance money click to claim now",
    "Forward to 10 friends and receive ₹500 Paytm cash",
    "Exclusive deal for you only 99% off limited time offer",
    "Your UPI account is limited please verify your identity now",
    "Your PhonePe account is blocked please reverify your KYC",
    "Adult content click here if you are over 18",
    "Your device has a virus! Download our free antivirus NOW",
    "Home loan at 0% interest rate limited time offer apply now",
    "Win a vacation to Maldives you have been selected winner",
]

HAM_MESSAGES = [
    "Hey, are we still on for lunch tomorrow at noon?",
    "Please find attached the quarterly report for your review",
    "Thanks for the meeting today. I will follow up on the action items",
    "Can you send me the file we discussed in the last call?",
    "Happy birthday! Hope you have a wonderful day",
    "The project deadline has been moved to next Friday",
    "I wanted to check in and see how the onboarding is going",
    "Could you review this pull request when you get a chance?",
    "Reminder: team standup at 9am tomorrow in conference room B",
    "Just saw your presentation, great work on the analysis",
    "The kids had a great time at the birthday party yesterday",
    "Can we reschedule our meeting to Thursday instead?",
    "Attached is the invoice for last month's services",
    "Looking forward to seeing you at the conference next week",
    "The flight is confirmed for departure at 6:45am",
    "Please complete the mandatory compliance training by Friday",
    "Your order has been shipped and will arrive in 3-5 business days",
    "Meeting notes from today's strategy session are attached",
    "The library book you requested is now available for pickup",
    "Thanks for submitting your application we will be in touch",
    "Your appointment is confirmed for Tuesday at 2:30pm",
    "Great game last night! That final quarter was intense",
    "The new feature is now deployed to production environment",
    "Could you proofread this document before I send it out?",
    "Just checking in to see if you need anything for the event",
    "Your package was delivered to the front door at 3:42pm",
    "The budget for Q3 has been approved by the finance team",
    "We are excited to welcome you to the team on Monday",
    "Please fill out the employee satisfaction survey by EOD",
    "We would love to have you over for dinner this weekend",
]


# ─────────────────────────────────────────
#  2. TEXT PREPROCESSING
# ─────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Clean and normalize raw message text.
    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove email addresses
      4. Remove phone numbers
      5. Replace currency symbols with token
      6. Remove special characters (keep alphanumeric + space)
      7. Collapse whitespace
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' PHONE ', text)
    text = re.sub(r'[\$£€₹\u20b9]', ' CURRENCY ', text)
    text = re.sub(r'[,]', '', text)          # strip commas (e.g. ₹1,00,000 → 100000)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────
#  3. DATASET PREPARATION
# ─────────────────────────────────────────

def build_dataset():
    messages = SPAM_MESSAGES + HAM_MESSAGES
    labels   = [1] * len(SPAM_MESSAGES) + [0] * len(HAM_MESSAGES)

    # Augment: keep original + lowercased + uppercased variant
    aug_msgs, aug_labels = [], []
    for msg, lbl in zip(messages, labels):
        aug_msgs.extend([msg, msg.lower(), msg.upper()])
        aug_labels.extend([lbl, lbl, lbl])

    df = pd.DataFrame({"text": aug_msgs, "label": aug_labels})
    df["clean_text"] = df["text"].apply(preprocess_text)
    return df


# ─────────────────────────────────────────
#  4. MODEL PIPELINE
# ─────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Pipeline:
      TF-IDF (unigrams + bigrams, sublinear TF, top-5000 features)
        → MLP with architecture [256 → 128 → 64]
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )),
        ("nn", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False,
        )),
    ])


# ─────────────────────────────────────────
#  5. TRAIN & EVALUATE
# ─────────────────────────────────────────

def train_and_evaluate(pipeline: Pipeline, df: pd.DataFrame):
    X = df["clean_text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training neural network…")
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "="*55)
    print("  SPAM CLASSIFIER — EVALUATION RESULTS")
    print("="*55)
    print(classification_report(y_test, y_pred, target_names=["Ham (0)", "Spam (1)"]))

    cm  = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("Confusion Matrix:")
    print(f"  {'':>10}  Pred Ham  Pred Spam")
    print(f"  {'True Ham':>10}  {cm[0,0]:^8}  {cm[0,1]:^9}")
    print(f"  {'True Spam':>10}  {cm[1,0]:^8}  {cm[1,1]:^9}")
    print(f"\nROC-AUC Score : {auc:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1")
    print(f"5-Fold CV F1  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("="*55)

    return pipeline


# ─────────────────────────────────────────
#  6. INFERENCE
# ─────────────────────────────────────────

def predict(pipeline: Pipeline, messages: list[str]) -> list[dict]:
    clean = [preprocess_text(m) for m in messages]
    preds  = pipeline.predict(clean)
    probas = pipeline.predict_proba(clean)

    results = []
    for msg, pred, prob in zip(messages, preds, probas):
        results.append({
            "message"     : msg,
            "prediction"  : "SPAM" if pred == 1 else "HAM",
            "confidence"  : f"{max(prob)*100:.1f}%",
            "spam_prob"   : f"{prob[1]*100:.1f}%",
            "ham_prob"    : f"{prob[0]*100:.1f}%",
        })
    return results


# ─────────────────────────────────────────
#  7. MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    df       = build_dataset()
    pipeline = build_pipeline()
    pipeline = train_and_evaluate(pipeline, df)

    test_cases = [
        "Congratulations! You've won a free iPhone. Click here to claim.",
        "Namaste, just wanted to confirm our 3pm meeting tomorrow.",
        "URGENT: Your SBI account has been compromised. Verify now!",
        "Can you send me the report before end of day?",
        "Earn ₹50,000 per week from home! No experience needed!",
        "Team lunch moved to Wednesday at 12:30 at the usual place.",
    ]

    print("\n  LIVE PREDICTIONS")
    print("="*55)
    for r in predict(pipeline, test_cases):
        tag = "🚨 SPAM" if r["prediction"] == "SPAM" else "✅ HAM "
        print(f"{tag}  [{r['confidence']}]  {r['message'][:55]}")
    print("="*55)
