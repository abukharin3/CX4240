import numpy as np

class LogisticRegression:
	'''
	Logister regression classifier for music lyrics. Given a set of lyrics, learn
	to classify which genre a song corresponds to.
	'''
	def __init__(self, data, genres):
		self.data = data # [genre,wordvector] 4000 songs 
		self.genres = genres
		self.vec_size = len(data[0][1])
		self.theta_list = []

	def learn_parameters(self):
		'''
		Gradient descent on the loss function. Creates a logistic regression model
		for each of the genres in genrelist. Appends each theta vector (which
		includes a bias) to the theta_list.
		'''
		for k in range(len(genres)):
			#initialize the theta vector
			theta = np.zeros(self.vec_size + 1) + 0.5
			#need to find a way to stop descent other than hardcoding
			count = 0
			while (count < 10):
				print("The count is " + str(count))
				loss_list = []
				#go thru each vector and calculate the loss, append to total list
				for i in range(len(self.data)):
					#include bias
					word_vec = np.insert(self.data[i][1],0,1)
					product = vector_multiplication(word_vec, theta)
					h_theta = sigmoid(product)
					loss = h_theta - (self.data[i][0] == genres[k])
					loss_list.append(loss)
				#adjust theta
				alpha = 0.01 #learning rate
				for j in range(len(theta)):
					print(j)
					total_loss = 0
					for i in range(len(self.data)):
						word_vec = np.insert(self.data[i][1],0,1)
						total_loss += (loss_list[i] * word_vec[j])
					theta[j] = theta[j] - (alpha * total_loss)
				#this is the makeshift way to stop the descent, add to the counter (will change later)
				count += 1
			self.theta_list.append(theta)

	def predict(self, song):
		'''
		Makes a prediction for the genre of the song. Must be bag
		of words form with the same vocabulary.
		'''
		song = np.array(song)
		#include bias
		song = np.insert(son,0,1)
		prediction_list = []
		#calculate sigmoid at each genre
		for i in range(len(theta_list)):
			product = vector_multiplication(song,theta_list[i])
			prediction_list.append(sigmoid(product))
		#returns the index of the genre list that has highest probability
		return prediction_list.index(max(prediction_list))

 

# Accesory functions to be used in the above class
def sigmoid(z):
	return (1 / (1 + np.exp(-1 * z)))

def vector_multiplication(vector_one, vector_two):
	total = 0
	for i in range(len(vector_one)):
		total += vector_one[i] * vector_two[i]
	return total



# Train the model
train_path = r"C:\Users\mwrep\OneDrive\Documents\CX_4240\training_songs.npy"
train_songs = np.load(train_path, allow_pickle = True)
genres = [0, 1, 2, 3, 4] # Country, pop, rap, rnb, rock

lr = LogisticRegression(data = train_songs, genres = genres)
lr.learn_parameters()

# Test the model
test_path = r"C:\Users\mwrep\OneDrive\Documents\CX_4240\test_songs.npy"
test_songs = np.load(test_path, allow_pickle = True)
success = 0
failure = 0
for i in range(len(test_songs)):
	if (lr.predict(test_songs[i][1]) == test_songs[i][0]):
		success += 1
	else:
		failure += 1

print(success/(success+failure))