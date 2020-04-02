import numpy as np
from matplotlib import pyplot as plt

train_path = r"C:\Users\mwrep\OneDrive\Documents\CX_4240\training_songs.npy"
train_songs = np.load(train_path, allow_pickle = True)

vec_list = []
for i in range(len(train_songs)):
	vec_list.append(train_songs[i][1])

matrix = np.array(vec_list)

#centering
total = np.zeros(len(matrix[0]))
for i in range(len(matrix)):
    total += matrix[i]
data_mean = total/len(matrix)
for i in range(len(matrix)):
    matrix[i] = matrix[i] - data_mean


#eigenstuff of covariance matrix
cov = np.matmul(matrix.T, matrix)
lamb, Q = np.linalg.eig(cov)

#find the highest eigenvalues
k = 40 #choose the number of columns to keep
'''
lamb_indices = []
for j in range(k):
    for i in range(len(lamb)):
        if ((lamb[i] == np.amax(lamb)) and (i not in lamb_indices)):
            temp_i = i
    lamb[temp_i] = np.amin(lamb)
    lamb_indices.append(temp_i)
'''
#make a matrix with strongest eigenvectors
Q_hat = np.empty((len(Q),k))
for i in range(len(Q)):
    count = 0
    for j in range(k):
        Q_hat[i][count] = Q[i][j]
        count += 1

#save the rank reduction matrix
np.save("rank_reduction_matrix.npy", Q_hat)

#create the reduced matrix
matrix_hat = np.matmul(matrix,Q_hat)

#do the same for the test songs
test_path = r"C:\Users\mwrep\OneDrive\Documents\CX_4240\test_songs.npy"
test_songs = np.load(test_path, allow_pickle = True)
test_vec_list = []
for i in range(len(test_songs)):
	test_vec_list.append(test_songs[i][1])
test_matrix = np.array(test_vec_list)
test_matrix_hat = np.matmul(test_matrix,Q_hat)

#unpack/repack the train matrix
data = []
for i in range(len(matrix_hat)):
    data.append(np.array([train_songs[i][0],matrix_hat[i]]))
data = np.array(data)

np.save("reduced_training_songs.npy", data)

#unpack/repack the test matrix
test_data = []
for i in range(len(test_matrix_hat)):
    test_data.append(np.array([test_songs[i][0],test_matrix_hat[i]]))
test_data = np.array(test_data)

np.save("reduced_test_songs.npy", test_data)