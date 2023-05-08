import numpy as np
import matplotlib.pyplot as plt
import torch as nn
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

#Importing the data and cleaning it
df = pd.read_csv('./spotify_songs.csv')
df = df.drop(df[df["language"]!="en"].index)
songs = df[['track_name','lyrics','playlist_genre']]
songs = songs.dropna()
songs = songs.reset_index(drop=True)

#Imports in the BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

lyrics_tokenized = songs['lyrics'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

lyrics_tensors = []
for i in lyrics_tokenized:
    lyrics_tensors.append(nn.tensor(i))

lyrics_tensors_padded = nn.pad_sequence(lyrics_tensors, batch_first=True, padding_value=0, max_len=512)

with nn.no_grad():
    embeddings = model(lyrics_tensors_padded)

#class Deep_Learning_Lyrics(nn.Module):
#    def __init__():
#
#   def train