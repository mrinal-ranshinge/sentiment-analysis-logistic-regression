# Sentiment Analysis using Logistic Regression

A text classification model that predicts whether a movie review expresses positive or negative sentiment.

## Objective
Sentiment analysis is one of the most common applied NLP tasks — used for product reviews, social media monitoring, and customer feedback analysis. This project builds an end-to-end sentiment classifier: from raw review text, to numerical features, to a trained prediction model.

## Dataset
A custom-built set of 30 short movie review sentences, manually labeled as positive (1) or negative (0), covering a mix of clearly positive, clearly negative, and moderately-worded reviews.

## Approach
1. Split the labeled review data into training (78%) and test (22%) sets.
2. Converted the raw text into numerical features using **TF-IDF vectorization**.
3. Trained a **Logistic Regression** classifier on the TF-IDF features.
4. Evaluated the model on the held-out test set.
5. Tested the trained model on new, unseen review sentences to confirm it generalizes beyond the training data.

## Tools
Python, scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `train_test_split`, `accuracy_score`), Matplotlib, NumPy

## Result
**85.71% accuracy** on the test set. The model correctly classified new sample reviews it had never seen before (e.g. correctly identifying "It was a masterfully crafted story" as positive and "The performance was terrible and dry" as negative), along with prediction confidence probabilities for each.

## Notes / Next Steps
- The dataset is small (30 reviews), which limits how well this generalizes — training on a larger labeled dataset (e.g. IMDB reviews) would give a more robust and realistic accuracy figure.
- Comparing Logistic Regression against other classifiers (e.g. Naive Bayes, which tends to do well on text) would be a good next experiment.
