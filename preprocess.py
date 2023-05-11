import numpy as np
import matplotlib.pyplot as plt
import torch as nn
import pandas as pd
import transformers
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Run BERT on GPU since its faster than CPU
device = nn.device('cuda' if nn.cuda.is_available() else 'cpu')

#Importing the data and cleaning it
df = pd.read_csv('./spotify_songs.csv')
df = df.drop(df[df["language"]!="en"].index)
songs = df[['lyrics','playlist_genre']]
songs = songs.dropna()
songs = songs.reset_index(drop=True)
num_genres = songs['playlist_genre'].nunique()

#Converts the musical genre strings into numerical values
le = LabelEncoder()
le.fit(df["playlist_genre"])
songs["genre_num"] = le.transform(songs["playlist_genre"])

num_genres = songs['playlist_genre'].nunique()

#Splitting the dataset into train, validation, test
songs_train, songs_val, train_label, val_label = train_test_split(songs['lyrics'], songs['genre_num'], test_size=0.001, random_state=5)
songs_val, songs_test, val_label, test_label = train_test_split(songs_val, val_label, test_size=0.5, random_state=5)

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')

def preprocess_func(data):
    return tokenizer.batch_encode_plus(data, truncation=True, padding = True, return_tensors="pt")

#Tokenizing the lyrics
#lyrics_train = preprocess_func(songs_train.tolist())
lyrics_dev = preprocess_func(songs_val.tolist())
#lyrics_test = preprocess_func(songs_test.tolist())
lyrics_dev = {k:nn.tensor(v).to(device) for k,v in lyrics_dev.items()}

with nn.no_grad():
#    hidden_train = model(**lyrics_train)
    hidden_val = model(**lyrics_dev)

#lyrics_train_embedd = hidden_train.last_hidden_state[:,0,:]
lyrics_dev_embedd = hidden_val.last_hidden_state[:,0,:]

#lyrics_train["label"] = train_label
#lyrics_val["label"] = val_label
#lyrics_test["label"] = nn.tensor(test_label.tolist())

