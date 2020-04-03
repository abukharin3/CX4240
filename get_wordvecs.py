import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import gc

'''
This script is meant to construct sequences of word vectors from song lyrics,
using a pre-trained word2vec model from
https://github.com/eyaler/word2vec-slim/blob/master/GoogleNews-vectors-negative300-SLIM.bin.gz
'''
# Load data
path_country = r"C:\Users\Alexander\Downloads\country_songs.txt"
path_pop = r"C:\Users\Alexander\Downloads\pop_songs.txt"
path_rap = r"C:\Users\Alexander\Downloads\rap_songs.txt"
path_rnb = r"C:\Users\Alexander\Downloads\rnb_songs.txt"
path_rock =  r"C:\Users\Alexander\Downloads\rock_songs.txt"
paths = [path_country, path_pop, path_rap, path_rnb, path_rock]

def clean(word):
    '''
    Removes all punctuation from a word
    '''
    alphabet = "ZXCVBNMASDFGHJKLQWERTYUIOP"
    new = ""
    for letter in word:
        if letter.upper() in alphabet:
            new += letter.upper()
    return new
model = KeyedVectors.load_word2vec_format(r"C:\Users\Alexander\Downloads\GoogleNews-vectors-negative300-SLIM.bin.gz", binary=True)
print(len(model["Hello"]))

# Turn songs into lists
song_list = []
song_list.append([])
song_index = 0
genre = -1
label = [0]
for path in paths:
    genre += 1
    # Read data
    with open(path, "r") as file:
        songs = file.readlines()

    # Create a list of songs
    for line in songs:
        lyric = line.strip().split(" ")
        if lyric == ["Start", "of", "song"]:
            label.append(genre)
            song_index += 1
            song_list.append([])
        # Process all words in a song
        for possible in lyric:
            word = clean(possible)
            if word:
                song_list[song_index].append(word)

# Find max length of a song
maxl = 0
for song in song_list:
    if len(song) > maxl:
        maxl = len(song)
song_vecs = []
labels = np.array([i // 100 for i in range(500)])
for num in range(5):
    print(num)
    for song in song_list[num*1000: num *1000 + 100]:
        song_seq = []
        for i in range(maxl):
            if i > len(song) - 1:
                song_seq.append(np.zeros(300))
            else:
                try:
                    song_seq.append(model[song[i]])
                except:
                    song_seq.append(np.zeros(300))
        song_vecs.append(np.array(song_seq))
gc.collect()
song_vecs = np.array(song_vecs)
np.save("wordvecs.npy", song_vecs)
np.save("wordvec_labels.npy", labels)
