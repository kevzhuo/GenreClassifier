import numpy as np
import matplotlib.pyplot as plt
import torch as nn
import pandas as pd
import transformers
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


#Importing the data and cleaning it
df = pd.read_csv('./spotify_songs.csv')
df = df.drop(df[df["language"]!="en"].index)
songs = df[['track_name','lyrics','playlist_genre']]
songs = songs.dropna()
songs = songs.reset_index(drop=True)
le = LabelEncoder()
le.fit(df["playlist_genre"])
songs["genre_num"] = le.transform(songs["playlist_genre"])

num_genres = songs['playlist_genre'].nunique()
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels= num_genres)

tokenized = tokenizer(songs["lyrics"][0:20].tolist(), padding=True, truncation=True, return_tensors="pt")
labels = nn.tensor(songs["genre_num"][0:20].tolist())
data_train = tokenized
data_train["labels"] = labels

tokenized = tokenizer(songs["lyrics"][21:30].tolist(), padding=True, truncation=True, return_tensors="pt")
labels = nn.tensor(songs["genre_num"][21:30].tolist())
data_test = tokenized
data_test["labels"] = labels

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = sklearn.metrics.f1_score(labels, preds)
    return {
      'f1': f1,
  }

training_args = TrainingArguments(output_dir='./results',num_train_epochs=3)
trainer = Trainer(model=model, args=training_args, train_dataset=data_train, eval_dataset=data_test, compute_metrics = compute_metrics)
#TODO: train not working
trainer.train()
print(trainer.eval())

#predicted_genre_names = le.inverse_transform(predicted_labels)