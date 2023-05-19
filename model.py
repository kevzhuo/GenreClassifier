import preprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from preprocess import lyric_train_embedding, lyric_test_embedding, songs_train, songs_test, multimodal_train_embedding, multimodal_test_embedding, le, audio_train_embedding, audio_test_embedding
import torch as nn

#Weighted f1_score used as the metric
def metric(predicted, true):
    result = f1_score(true, predicted, average= "weighted")
    return result

#Multiclass Logistic Regression Model
def multiclass_logreg(X_train, y_train, X_test):
    classifier = LogisticRegression(max_iter = 1000)
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test)


def main():
    #For a baseline, we are predicting "pop" for every genre since it occurs the most
    pop = [2] * len(songs_test["genre_num"])
    print("Baseline: " + str(metric(pop, songs_test["genre_num"])))
    lyric_only_results = multiclass_logreg(lyric_train_embedding, songs_train["genre_num"].tolist(), lyric_test_embedding)
    print("Lyric Only: " + str(metric(lyric_only_results, songs_test["genre_num"].tolist())))
    multimodal_results = multiclass_logreg(multimodal_train_embedding, songs_train["genre_num"].tolist(), multimodal_test_embedding)
    print("Multimodal: " + str(metric(multimodal_results, songs_test["genre_num"].tolist())))
    audio_results = multiclass_logreg(audio_train_embedding, songs_train["genre_num"].tolist(), audio_test_embedding)
    print("Audio: " + str(metric(audio_results, songs_test["genre_num"].tolist())))

if __name__ == '__main__':
    main()