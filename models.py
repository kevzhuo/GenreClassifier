import preprocess
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score
import torch as nn

#Weighted f1_score used as the metric
def metric(predicted, true):
    f1 = f1_score(true, predicted, average = 'weighted')
    return f1

#Lyric only based BERT model, calculates most likely genre for each song and outputs a list
def lyric_genre(test_set):
    model = BertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels= preprocess.num_genre)
    results = []
    for song in test_set:
        with nn.no_grad():
            logits = model(**song).logits
        predicted_class_id = logits.argmax().item()
        results.append(predicted_class_id)
    return results

#Multimodal BERT model that uses lyrics + acoustic features of the song
def multimodal(test_set):
    pass

def main():
    #For a baseline, we are predicting "pop" for every genre since it occurs the most
    pop = [2] * len(preprocess.test_label)
    print("Baseline: " + str(metric(pop, preprocess.test_label)))
    #print("Lyric Only: " + metric(lyric_genre(preprocess.lyric_test), preprocess.test_label))

if __name__ == '__main__':
    main()