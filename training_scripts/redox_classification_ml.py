import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

num_pcs = 5


# Read data
OXPATH = '/Users/peterdahl/Documents/BatistaLab/heme_pes/build_bis_imid/oxidized/production/energy_gaps/machine_learning/trajectories/oxidized_pca/rerun/'
ox_proj = np.loadtxt(OXPATH+"pca_proj.txt", delimiter = " ")
ox_labels = np.zeros(np.shape(ox_proj)[0])

REDPATH = '/Users/peterdahl/Documents/BatistaLab/heme_pes/build_bis_imid/oxidized/production/energy_gaps/machine_learning/trajectories/reduced_pca/red_on_ox/'
red_proj = np.loadtxt(REDPATH+"pca_proj_red.txt", delimiter = " ")
red_labels = np.ones(np.shape(red_proj)[0])

tot_data = np.append(ox_proj,red_proj,axis=0)
tot_data = tot_data[:,0:num_pcs]
tot_labels = np.append(ox_labels,red_labels)


# split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(tot_data, tot_labels, test_size=0.30, random_state=40)
#print(y_test[0:10])

# the labels (defined as 0 for oxidized and 1 for reduced) need to be encoded as categorical variables
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#print(y_test[0:10])

def ml(X_train,X_test,y_train,y_test):
	global test_prediction

	model = Sequential()
	model.add(Dense(500, activation='relu', input_dim=num_pcs))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(2, activation='softmax'))

	# Compile the model
	model.compile(optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy'])

	# build the model
	model.fit(X_train, y_train, epochs=25)

	#model.save("model_1pcs")

	training_prediction = model.predict(X_train)
	scores = model.evaluate(X_train,y_train,verbose=0)
	print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

	test_prediction = model.predict(X_test)
	scores2 = model.evaluate(X_test,y_test,verbose=0)
	print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))

	

ml(X_train, X_test, y_train, y_test)

#def redox(cat_val):
#	if int(cat_val[0]) == 0:
#		return 'reduced'
#	elif int(cat_val[0]) == 1:
#		return 'oxidized'

fileOUT = open('ml_results.txt','w')
for i in range(len(X_test)):
	print('%s %s' % (y_test[i],test_prediction[i]), file=fileOUT)
