import os
import numpy as np
import pandas as pd
import pickle as pkl


if __name__ == '__main__':
	root = '/scratch/groups/willhies/echo_view_classifier/'
	pred_dir = os.path.join(root, 'predictions')
	csv_path = os.path.join(pred_dir, 'results_with_probs.csv')

	truth_list = []
	pred_list = []
	filenames = []
	A4C_probs = []
	PLAX_probs = []
	PSAX_probs = []
	
	with open(csv_path) as fp:
		line = fp.readline()
		counter = 0
		while line:
			if counter == 0:
				counter += 1
				line = fp.readline()
				continue
			else:
				line = line.strip('\n')
				items = line.split(',')
				subitems = items[0].split('/')
				truth = subitems[0]
				pred = items[1]
				filename = subitems[1]
				correct_bool = (truth == pred)
				A4C_prob = items[2]
				PLAX_prob = items[3]
				PSAX_prob = items[4]
				if correct_bool:
					print('Filename:', filename, 'Truth:', truth, 'Pred:', pred, 'Bool:', correct_bool)
					truth_list.append(truth)
					pred_list.append(pred)
					filenames.append(filename)
					A4C_probs.append(A4C_prob)
					PLAX_probs.append(PLAX_prob)
					PSAX_probs.append(PSAX_prob)
				line = fp.readline()
				counter += 1

	rights_path_list = []
	test_dir = '/scratch/groups/willhies/echo_view_classifier/dataset/test/'
	for f,truth in zip(filenames, truth_list):
		rights_path_list.append(os.path.join(os.path.join(test_dir, truth), f))

	pkl_path = os.path.join(pred_dir, 'rights_path_list.pkl')
	pkl.dump(rights_path_list, open(pkl_path, 'wb'))

	rights = pd.DataFrame({'Filename':filenames, 'Truth':truth_list, 'Predictions':pred_list, 'A4C Prob':A4C_probs, 'PLAX Prob':PLAX_probs, 'PSAX Prob':PSAX_probs})
	rights.to_csv(os.path.join(pred_dir, 'rights.csv'), index=False)
