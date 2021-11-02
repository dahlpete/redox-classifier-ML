# redox-classifier-ML
Machine Learning approaches to predicting the redox state of a heme from its geometrical fluctuations

This repository contains three ML models used to predict the redox state of a c-type heme from its geometry: 
  1) Linear classifier
  2) Neural network classifier
  3) Neural network for predicting vertical detachment energies

All three models are trained on geometric descriptions of oxidized and reduced hemes provided by an 
essential model analysis (principle component analysis) of hemes simulated using the CHARMM36 force field.
The molecular dynamics were run on a truncated heme system containing the heme, and the covalently bonded
HIS and CYS residues (trunctated at the beta carbon).

The classifier models take in the projections along the essential modes (principle components) for known 
redox states. The models are built using a user defined number of essential modes. The user can then use 
the redox_classifier_application.py script to pass projections from any other simulation of a c-type heme
through the model to get a probability of the heme being in an oxidized or reduced state.

When trained with at least 20 essential modes, the classifier is more than 95% accurate in classifying the 
test data. However, a classifier model appears to be too simplistic to consistently predict the redox state 
of hemes simulated in a protein environment.

The more sophisticated model in this repository is a neural network used to predict vertical detachment 
energies (ionization energies) from hemes, based on their geometry. Vertical detachment energies can be used
to directly compute a heme redox potential using the linear response approximation. This model is trained on 
the same essential mode analysis as the classifier model, but is paired with DFT level calculations of vertical
detachment energies. These calculations are performed at an omegaB97X-D level of theory with cc-pVDZ basis sets
for non-FE atoms and a VDZ basis set for FE.

In its current state, this model is capable of matching general trends in the energies as a function of the 
essential modes. However, when the predicted energies are averaged over the test set, there still exists about 
a 150 meV error relative to the DFT calculated energies.
