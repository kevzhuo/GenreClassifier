import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pandas as pd
import transformers
from transformers import BertTokenizer, BertModel
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Importing the data and cleaning it
df = pd.read_csv('./spotify_songs.csv')
df = df.sample(n=100, random_state = 50)
df = df.drop(df[df["language"]!="en"].index)
songs = df[['lyrics','playlist_genre', 'acousticness', 'tempo', 'valence', 'danceability', 'energy']]
songs = songs.dropna()
songs = songs.reset_index(drop=True)
num_genres = songs['playlist_genre'].nunique()

  
#Converts the musical genre strings into numerical values
le = LabelEncoder()
le.fit(df["playlist_genre"])
songs["genre_num"] = le.transform(songs["playlist_genre"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = transformers.BertTokenizer.from_pretrained("distilbert-base-uncased")
model = transformers.BertModel.from_pretrained("distilbert-base-uncased").to(device)

songs_train, songs_test = train_test_split(songs, test_size=0.3, random_state=5)

tokenized_train = tokenizer(songs_train["lyrics"].values.tolist(), max_length = 100, padding = True, truncation = True, return_tensors="pt")
tokenized_test = tokenizer(songs_test["lyrics"].values.tolist() , max_length = 100, padding = True, truncation = True,  return_tensors="pt")

#move to GPU since it is faster
tokenized_train = {k:torch.tensor(v).to(device) for k,v in tokenized_train.items()}
tokenized_test = {k:torch.tensor(v).to(device) for k,v in tokenized_test.items()}

#Gets the embeddings
with torch.no_grad():
  hidden_train = model(**tokenized_train) 
  hidden_test = model(**tokenized_test)

cls_train = hidden_train.last_hidden_state[:,0,:]
cls_test = hidden_test.last_hidden_state[:,0,:]

lyric_train_embedding = cls_train.to("cpu")
lyric_test_embedding = cls_test.to("cpu")

audio_train_embedding = torch.cat((torch.unsqueeze(torch.Tensor(songs_train['acousticness'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_train['energy'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_train['tempo'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_train['valence'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_train['danceability'].tolist()),1)), -1)
audio_test_embedding = torch.cat((torch.unsqueeze(torch.Tensor(songs_test['acousticness'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_test['energy'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_test['tempo'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_test['valence'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_test['danceability'].tolist()),1)), -1)

#Concatenating the acoustic vectors onto the lyric BERT embedding
multimodal_train_embedding = torch.cat((torch.unsqueeze(torch.Tensor(songs_train['acousticness'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_train['energy'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_train['tempo'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_train['valence'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_train['danceability'].tolist()),1), lyric_train_embedding),-1)
multimodal_test_embedding = torch.cat((torch.unsqueeze(torch.Tensor(songs_test['acousticness'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_test['energy'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_test['tempo'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_test['valence'].tolist()),1), torch.unsqueeze(torch.Tensor(songs_test['danceability'].tolist()),1), lyric_test_embedding), -1)
