Hate Speech Detection — Machine Learning Backend (TF-IDF + Logistic Regression)

This project classifies tweets as:

0 → Normal

1 → Hate Speech / Offensive / Abusive

It uses:

Text cleaning + preprocessing

TF-IDF vectorization

Logistic Regression

Predict function for inference

No frontend is included — this is a backend-only ML system.

🚀 Features

Cleans tweets (URLs, mentions, hashtags, punctuation, stopwords)

Converts text to TF-IDF features

Trains a Logistic Regression classifier

Provides a function predict_tweet(text) for inference

📊 Dataset

Your code loads:

tweet_df = pd.read_csv("train.csv")


Dataset must contain:

column	description
tweet	raw tweet text
label	0 or 1
🧹 Preprocessing

Defined in:

def data_processing(text):
    ...


Includes:

Lowercasing

Remove URLs

Remove mentions

Keep hashtag word

Remove punctuation

Remove stopwords (sklearn + custom: {"rt","amp"})

Regex tokenization

Remove short words

This creates clean text ready for TF-IDF.

🧠 Model Used
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2)
clf = LogisticRegression(max_iter=300, solver='liblinear')

📈 Accuracy & Evaluation
accuracy_score(y_test, pred)
classification_report(y_test, pred)
ConfusionMatrixDisplay.from_predictions()

🔮 Prediction Function (Important)

Included in your code:

def predict_tweet(text, threshold=0.5):
    clean = data_processing(text)
    X = tfidf.transform([clean])
    prob1 = clf.predict_proba(X)[0, 1]
    pred = int(prob1 >= threshold)
    return {"input": text, "cleaned": clean, "pred": pred, "prob_class1": prob1}


Prediction Output Example:

predict_tweet("@user thanks for #lyft credit i can't use… #disappointed")


Returns something like:

{
  "input": "...",
  "cleaned": "thanks lyft credit cant use disappointed",
  "pred": 0,
  "prob_class1": 0.18
}


📦 Installation
pip install numpy pandas scikit-learn matplotlib nltk

▶ Train Model
python train.py
