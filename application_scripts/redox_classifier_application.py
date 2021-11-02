import numpy as np
import keras
import sys

my_model = sys.argv[1]
my_data = sys.argv[2]
n_pcs = int(sys.argv[3])

my_model = keras.models.load_model(my_model)
projections = np.loadtxt(my_data)[:,:n_pcs]

predictions = my_model.predict(projections)
p_ox = np.mean(predictions[:,0])
p_red = np.mean(predictions[:,1])

fileOUT = open('model_predictions.txt','w')
print('P(ox)\tP(red)',file=fileOUT)
for p in predictions:
	print('%.3f\t%.3f' % (p[0],p[1]),file=fileOUT)

print('P(ox) = %.3f' % p_ox)
print('P(red) = %.3f' % p_red)

fileOUT.close()
