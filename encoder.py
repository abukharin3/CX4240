import numpy as np

class encoder:
    '''
    Given a set of tweets, this class can create a vocabulary
    and encode the tweets as a bag of words
    '''
    def __init__(self, data):
        self.data = self.transform(data) # array [song, label]
        np.random.shuffle(self.data)
        self.vocab = self.get_vocab()
        self.data = self.n_gram(data)

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

    def discretize(self, sentence):
        # Turns a sentence into a list of words
        sentence_list = []
        for word in sentence.split(" "):
            sentence_list.append(self.clean(word))
        return sentence_list

    def n_gram(self, data):
        new = []
        for i in range(len(data)):
            sentence = data[i][0]
            new.append(np.array([self.encode(sentence), data[i][1]]))
        return(np.array(new))


    def encode(self, sentence):
        '''
        Method to encode a song using bag of word model. song must be
        a list of words.
        '''
        if "<unk>" not in self.vocab:
            self.vocab.append("<unk>")
        vec = np.zeros(len(self.vocab))
        for word in sentence:
            if word.upper() in self.vocab:
                index = self.vocab.index(word.upper())
                vec[index] += 1
            else:
                index = self.vocab.index("<unk>")
                vec[index] += 1
        print(vec)
        return np.array(vec)

    def get_vocab(self):
        # Get vocab from only the training data
        length = len(self.data)
        train = self.data[0 : length // 10 * 8]
        all_words = []
        for i in range(len(train)):
            for word in self.data[i][0]:
                all_words += [self.clean(word)]
        return list(set(all_words))

    def transform(self, data):
        new = []
        for i in range(len(data)):
            sentence = data[i][0]
            new.append(np.array([sentence, data[i][1]]))
        return np.array(new)

    def save(self):
        length = len(self.data)
        train = self.data[0 : length // 10 * 8]
        vocab = np.array(self.vocab)
        test = self.data[length // 10 * 8 :]
        np.save("train_songs.npy", train)
        np.save("test_songs.npy", test)

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
            song_index += 1
            song_list.append(np.array([[], genre]))
        elif len(lyric[0]) > 0:
            for word in lyric:
                song_list[song_index][0].append(word)
    genre += 1
song_list = song_list[1:]
e = encoder(np.array(song_list))
e.save()





# test_data = []
# for song in song_list:
#     if len(song) > 0:
#         genre = song[0]
#         test_data.append(np.array([genre, encoder.encode(song[1])]))
# test_data = np.array(test_data)

# np.save("test_songs.npy", test_data) # [num songs, genre, vec]



