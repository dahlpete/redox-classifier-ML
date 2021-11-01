import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split

num_pcs = 50

OXPATH = '/Users/peterdahl/Documents/BatistaLab/heme_pes/build_bis_imid/oxidized/production/energy_gaps/machine_learning/trajectories/oxidized_pca/rerun/'
ox_proj = np.loadtxt(OXPATH+"pca_proj.txt", delimiter = " ")
ox_labels = np.zeros(np.shape(ox_proj)[0])

REDPATH = '/Users/peterdahl/Documents/BatistaLab/heme_pes/build_bis_imid/oxidized/production/energy_gaps/machine_learning/trajectories/reduced_pca/red_on_ox/'
red_proj = np.loadtxt(REDPATH+"pca_proj_red.txt", delimiter = " ")
red_labels = np.ones(np.shape(red_proj)[0])

tot_data = np.append(ox_proj,red_proj,axis=0)
tot_data = tot_data[:,0:num_pcs]
tot_labels = np.append(ox_labels,red_labels)

X_train, X_test, y_train, y_test = train_test_split(tot_data, tot_labels, test_size=0.30, random_state=40)


pc_features = [tf.feature_column.numeric_column(str(j)) for j in range(1,num_pcs+1)]

model = tf.estimator.LinearClassifier(n_classes=2,model_dir='ongoing/model1',feature_columns=pc_features)

FEATURES = [str(j) for j in range(1,num_pcs+1)]
LABEL='label'

def get_input_fn(data_set,labels,training=False,batch_size=256):
	# input function is a dictionary relating features to data
	features = {k: data_set[:,int(k)-1] for k in FEATURES}
	dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))

	if training:
		dataset=dataset.shuffle(1000).repeat()

	return dataset.batch(batch_size)

model.train(input_fn=lambda:get_input_fn(X_train,y_train,training=True),steps=5000)

test_data = model.evaluate(input_fn=lambda:get_input_fn(X_test,y_test),steps=5000)

print('\n linear analysis:\n')
print(test_data)
print('\n')
