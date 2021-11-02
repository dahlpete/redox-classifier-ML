import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
import sys
import csv

# Load PCA results from oxidized heme model
evectors = np.loadtxt('oxidized_pca/evectors.txt')

# Load the trajectory you wish to analyze
structure_file = sys.argv[1] # This is the .psf file
trajectory = sys.argv[2]
selection = "resid 410 and segid Z or (resid 47 50 and segid B and not name C O N HN HA) or (resid 2 51 and segid B and not name CA C O N HN HA)"

u = mda.Universe(structure_file,trajectory)
atomselection = u.select_atoms(selection)


ref_u = mda.Universe('myprot_reference_ox.psf','myprot_reference_ox.pdb')
canvas_u = mda.Universe('myprot_reference_ox.psf','myprot_reference_ox.pdb')

def evec_proj(evec,coords):
	proj = np.zeros(np.shape(evec)[1])
	for i in range(np.shape(evec)[1]):
		proj[i] = np.dot(evec[:,i],coords)

	return proj 

def h_coords(cap1,cap2,bond_length=1.10):
	coords1 = cap1.positions.ravel()
	coords2 = cap2.positions.ravel()

	veclength = np.sqrt(np.sum((coords2 - coords1)**2))
	h_xyz = coords1 + (coords2 - coords1) * (bond_length/veclength)

	return h_xyz

def get_caps(atm_grp,res):
	if res == 'axial':
		# determine positions of capping hydrogens for axial ligands
		axial_lig = atm_grp.select_atoms('resname HSO HSR').residues
		new_h = [[] for i in range(len(axial_lig))]
		cnt = 0
		for ix in axial_lig.resids:
			seltext1 = 'resid %s and name CG' % ix
			cap1 = atm_grp.select_atoms(seltext1)

			seltext2 = 'resid %s and name CB' % ix
			cap2 = atm_grp.select_atoms(seltext2)

			new_h[cnt] = list(h_coords(cap1,cap2))
			cnt += 1

	elif res == 'thio':
		thio_lig = atm_grp.select_atoms('resname CYO CYR').residues
		new_h = [[] for i in range(len(thio_lig))]
		cnt = 0
		for ix in thio_lig.resids:
			seltext1 = 'resid %s and name CB' % ix
			cap1 = atm_grp.select_atoms(seltext1)

			seltext2 = 'resid %s and name CA' % ix
			cap2 = atm_grp.select_atoms(seltext2)

			new_h[cnt] = list(h_coords(cap1,cap2))
			cnt += 1

	return new_h


def coordinate_mapping(atm_grp,ref_grp,canvas_grp,rid_dict):
	# This function maps the coordinates of the atomgroup of interest onto a canvas universe
	# The canvas universe is a copy of the reference universe. By mapping to the canvas universe
	# we enable easy alignment and further analysis.

	ref_res = ref_grp.resnames
	ref_rid = ref_grp.resids
	ref_names = ref_grp.names

	coordinates = np.zeros([len(ref_res),3])
	for i in range(len(ref_res)):
		res = ref_res[i]; rid = ref_rid[i]; name = ref_names[i]
		
		atm_rid = rid_dict[rid]
		if name == 'HG1':
			# Check which of the hydrogen coords are closest to the gamma carbon of the same residue
			hydrogen_coords = get_caps(atm_grp,res='axial')
			c_gamma = atm_grp.select_atoms('resid '+str(atm_rid)+' and name CG').positions.ravel()
			cnt = 0
			dist = np.zeros(len(hydrogen_coords))
			for h in hydrogen_coords:
				dist[cnt] = np.sqrt(np.sum(np.array([(h[j]-c_gamma[j])**2 for j in range(len(h))])))
				cnt += 1
			if dist[0] < dist[1]:
				coordinates[i,:] = hydrogen_coords[0]
			else:
				coordinates[i,:] = hydrogen_coords[1]

		elif name == 'HB3':
			hydrogen_coords = get_caps(atm_grp,res='thio')
			c_beta = atm_grp.select_atoms('resid '+str(atm_rid)+' and name CB').positions.ravel()
			cnt = 0
			dist = np.zeros(len(hydrogen_coords))
			for h in hydrogen_coords:
				dist[cnt] = np.sqrt(np.sum(np.array([(h[j]-c_beta[j])**2 for j in range(len(h))])))		
				cnt += 1
			if dist[0] < dist[1]:
				coordinates[i,:] = hydrogen_coords[0]
			else:
				coordinates[i,:] = hydrogen_coords[1]
		else:
			sel = atm_grp.select_atoms('resid '+str(atm_rid)+' and name '+name)
			coordinates[i,:] = sel.positions.ravel()

	canvas_grp.positions = coordinates

	return 0


def alignment_indices(atm_grp,ref_grp,canvas_grp):
	ref_res = ref_grp.resnames
	ref_rid = ref_grp.resids
	ref_names = ref_grp.names

	# which axial ligands map to which axial ligands
	ref_axlig_rid = np.unique(ref_grp.select_atoms('resname IMO IMR HSO HSR').resids)
	atm_axlig_rid = np.unique(atm_grp.select_atoms('resname IMO IMR HSO HSR').resids)

	axdict1 = {ref_axlig_rid[i]:atm_axlig_rid[i] for i in range(len(ref_axlig_rid))}

	atm_axlig_rid = atm_axlig_rid[::-1] #np.array(list(atm_axlig_rid).reverse())
	axdict2 = {ref_axlig_rid[i]:atm_axlig_rid[i] for i in range(len(ref_axlig_rid))}

	# which cys ligands map to which cys ligands
	ref_cyslig_rid = np.unique(ref_grp.select_atoms('resname COB CRB CYO CYR').resids)
	atm_cyslig_rid = np.unique(atm_grp.select_atoms('resname COB CRB CYO CYR').resids)

	cysdict1 = {ref_cyslig_rid[i]:atm_cyslig_rid[i] for i in range(len(ref_cyslig_rid))}

	atm_cyslig_rid = atm_cyslig_rid[::-1] #np.array(list(atm_cyslig_rid).reverse())
	cysdict2 = {ref_cyslig_rid[i]:atm_cyslig_rid[i] for i in range(len(ref_cyslig_rid))}

	# heme mapping
	ref_hme_rid = np.unique(ref_grp.select_atoms('resname HEO HER').resids)
	atm_hme_rid = np.unique(atm_grp.select_atoms('resname HEO HER').resids)
	
	hmdict = {ref_hme_rid[i]:atm_hme_rid[i] for i in range(len(ref_hme_rid))}

	dictlist =     [{**axdict1,**cysdict1,**hmdict},
			{**axdict2,**cysdict1,**hmdict},
			{**axdict1,**cysdict2,**hmdict},
			{**axdict2,**cysdict2,**hmdict}]

	rmsd_list = list(np.zeros(len(dictlist)))
	count = 0
	for d in dictlist:
		coordinate_mapping(atm_grp,ref_grp,canvas_grp,d)
		[old_rmsd,rmsd_list[count]] = align.alignto(canvas_grp,ref_grp)
		count += 1
	
	dict_idx = rmsd_list.index(np.min(np.array(rmsd_list)))
	doi = dictlist[dict_idx]

	return doi

def create_and_project(universe,ref_grp,canvas_grp,rid_dict):
	n_frames = len(universe.trajectory)	
	proj = np.zeros([n_frames,np.shape(evectors)[1]])
	for i,ts in enumerate(universe.trajectory):
		atomselection = universe.select_atoms(selection)
		coordinate_mapping(atomselection,ref_grp,canvas_grp,rid_dict)

		# align canvas group with updated coordinates to the reference group
		rmsd_data = align.alignto(canvas_grp,ref_grp)
		new_coords = canvas_grp.positions.ravel()

		# project the aligned coordinates on the eigenvectors
		proj[i,:] = evec_proj(evectors,new_coords)
	canvas_grp.write('aligned_struct.pdb')

	return proj
	

	
def main():
	doi = alignment_indices(atomselection,ref_u.select_atoms('all'),canvas_u.select_atoms('all'))
	proj_array = create_and_project(u,ref_u.select_atoms('all'),canvas_u.select_atoms('all'),doi)

	with open('atomselect_proj.txt','w') as f:
		csv_writer=csv.writer(f,delimiter=' ')
		csv_writer.writerows(proj_array)

main()


#model = keras.models.load_model(my_model)
