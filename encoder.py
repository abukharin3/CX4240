import numpy as np

class encoder:
    '''
    Given a song and previous vocabulary, this class will encode
    the song as a vector
    '''
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary # List of words

    def clean(self, word):
        '''
        Removes all punctuation from a word
        '''
        alphabet = "ZXCVBNMASDFGHJKLQWERTYUIOP1234567890"
        new = ""
        for letter in word:
            if letter.upper() in alphabet:
                new += letter.upper()
        return new

    def encode(self, song):
        '''
        Method to encode a song using bag of word model. song must be
        a list of words.
        '''
        vec = np.zeros(len(self.vocabulary))
        for possible in song:
            word = self.clean(possible)
            if word in self.vocabulary:
                index = self.vocabulary.index(word)
                vec[index] += 1
            else:
                index = self.vocabulary.index("<unk>")
                vec[index] += 1
        return np.array(vec)

# Now load data and get the test data
path_country = r"C:\Users\Alexander\Downloads\country_songs.txt"
path_pop = r"C:\Users\Alexander\Downloads\pop_songs.txt"
path_rap = r"C:\Users\Alexander\Downloads\rap_songs.txt"
path_rnb = r"C:\Users\Alexander\Downloads\rnb_songs.txt"
path_rock = r"C:\Users\Alexander\Downloads\rock_songs.txt"
paths = [path_country, path_pop, path_rap, path_rnb, path_rock]

song_list = []
song_list.append([])
song_index = 0
genre = 0
for path in paths:
    num_songs = 0
    # Read data
    with open(path, "r") as file:
        songs = file.readlines()

    # Create a list of songs
    for line in songs:
        lyric = line.strip().split(" ")
        if lyric == ["Start", "of", "song"]:
            num_songs += 1
        if num_songs > 800:
            if lyric == ["Start", "of", "song"]:
                song_index += 1
                song_list.append((genre, []))
            for word in lyric:
                song_list[song_index][1].append(word)
    genre += 1

vocabulary_path = r"C:\Users\Alexander\Downloads\training_vocab.txt"
with open(vocabulary_path, "r") as file:
    vocab = file.readline()

vocab = vocab.split(" ")
vocab = vocab[:-1]
encoder = encoder(vocab)

test_data = []
for song in song_list:
    if len(song) > 0:
        genre = song[0]
        test_data.append(np.array([genre, encoder.encode(song[1])]))
test_data = np.array(test_data)

np.save("test_songs.npy", test_data) # [num songs, genre, vec]



