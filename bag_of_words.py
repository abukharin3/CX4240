import numpy as np
import matplotlib.pylab as plt
# Load data
path_country = r"C:\Users\Alexander\Downloads\country_songs.txt"
path_pop = r"C:\Users\Alexander\Downloads\pop_songs.txt"
path_rap = r"C:\Users\Alexander\Downloads\rap_songs.txt"
path_rnb = r"C:\Users\Alexander\Downloads\rnb_songs.txt"
path_rock = r"C:\Users\Alexander\Downloads\rock_songs.txt"
paths = [path_country, path_pop, path_rap, path_rnb, path_rock]



def clean(word):
    '''
    Removes all punctuation from a word
    '''
    alphabet = "ZXCVBNMASDFGHJKLQWERTYUIOP1234567890"
    new = ""
    for letter in word:
        if letter.upper() in alphabet:
            new += letter.upper()
    return new

def n_gram(song, v_dict):
    '''
    Turns a text into a vector
    '''
    vec = np.zeros(len(v_dict.keys()))
    v_list = list(v_dict.keys())
    for word in song:
        if word in v_list:
            index = v_list.index(word)
            vec[index] += 1
        else:
            index = v_list.index("<unk>")
            vec[index] += 1
    return np.array(vec)


vocab_dict = {}
song_list = []
song_list.append([])
song_index = 0
for path in paths:

    # Read data
    with open(path, "r") as file:
        songs = file.readlines()

    # Create a list of songs
    for line in songs:
        lyric = line.strip().split(" ")
        if lyric == ["Start", "of", "song"]:
            song_index += 1
            song_list.append([])
            # Use only the first 800 out of 1000 songs for each genre
            if song_index % 800 == 0:
                break
        # Process all words in a song
        for possible in lyric:
            word = clean(possible)
            song_list[song_index].append(word)
            if word in vocab_dict.keys():
                vocab_dict[word] += 1
            else:
                vocab_dict[word] = 1


# Get rid of words with only one occurence
new_dict = {"<unk>": 0}
for word in vocab_dict.keys():
    if vocab_dict[word] <= 4:
        new_dict["<unk>"] += 1
    else:
        new_dict[word] = vocab_dict[word]
vocab_list = list(new_dict.keys())
# x = sorted(new_dict, key = lambda x : new_dict[x])
# print([(a, new_dict[a]) for a in x])

#Turn song list into a matrix
data = []
count = 0
for song in song_list:
    genre = count // 800
    count += 1
    if len(song) > 1:
        data.append(np.array([genre, n_gram(song, new_dict)]))
data = np.array(data)

np.save("training_songs.npy", data) # [num songs, genre, vec]
for i in range(len(vocab_list)):
    vocab_list[i] = vocab_list[i] + " "
with open("training_vocab.txt", "w") as file:
    file.writelines(vocab_list)


