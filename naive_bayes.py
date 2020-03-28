import numpy as np

class NaiveBayes:
    '''
    Naive Bayes classifier for music lyrics. Given a set of lyrics, learn
    to classify which genre a song corresponds to. We make the assumption that
    all genres share the same prior distribution.
    '''
    def __init__(self, data, genres):
        self.data = data #[num songs (4000), [genre, wordvec]]
        self.genres = genres
        self.vocab_len = len(self.data[0][1])
        self.probs = self.learn_parameters() # [genres, vocabulary]
        self.count = self.get_count()

    def learn_parameters(self):
        '''
        Method to learn the posterior probabilities, probability of a word given
        a genre. Note: We have an equal number of songs from each class so we
        don't have to worry about the prior.
        '''
        probs = np.ones([len(self.genres), self.vocab_len])
        for genre in self.genres:
            for song in self.data[800 * genre : 800 * (genre + 1)]:
                probs[genre] += song[1] / sum(song[1])

        probs *= 1 / (800 + self.vocab_len)
        return probs

    def get_count(self):
        counts = np.sum(self.probs, axis = 0) * (800 + self.vocab_len)
        return counts


    def predict(self, song):
 # Song should be in bag of words form with the same vocab
        song = np.array(song)
        category = np.zeros(len(self.genres))
        for j in self.genres:
            for i in range(len(song)):
                category[j] += song[i] * self.probs[j][i]
        category = list(category)
        arg_max = category.index(max(category))
        return arg_max



# Train the model
train_path = r"C:\Users\Alexander\Documents\training_songs.npy"
train_songs = np.load(train_path, allow_pickle = True)
genres = [0, 1, 2, 3, 4] # Country, pop, rap, rnb, rock

nb = NaiveBayes(data = train_songs, genres = genres)
nb.learn_parameters()

#Test the model
test_path = r"C:\Users\Alexander\Downloads\test_songs.npy"

#Load test data
test_songs = np.load(test_path, allow_pickle = True)

success = [0,0,0,0,0]
total = 0
for i in test_songs:
    total += 1
    if i[0] == nb.predict(i[1]):
        success[i[0]] += 1
print(np.array(success) / 200)
print(sum(success) / 1000)
