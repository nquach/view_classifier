import os
from shutil import copyfile
import multiprocessing as mp
from utils import *

def filename2path(frames_dir):
	A4C_dir = os.path.join(frames_dir, 'A4C_frames')
	PLAX_dir = os.path.join(frames_dir, 'PLAX_frames')
	PSAX_dir = os.path.join(frames_dir, 'PSAX_frames')

	name2path = {}

	for f in os.listdir(A4C_dir):
		if f[-3:] == 'png':
			name2path[f] = os.path.join(A4C_dir, f)

	for f in os.listdir(PLAX_dir):
		if f[-3:] == 'png':
			name2path[f] = os.path.join(PLAX_dir, f)

	for f in os.listdir(PSAX_dir):
		if f[-3:] == 'png':
			name2path[f] = os.path.join(PSAX_dir, f)

	return name2path



if __name__ == '__main__':
	root = '/scratch/groups/willhies/echo_view_classifier/'

	source_direc = '/scratch/groups/willhies/echo_view_classifier/frames'

	vis_direc = os.path.join(root, 'guided_backprop_vis_RIGHTS')
	
	frame_direc = os.path.join(vis_direc, 'greyscale')
	safe_makedir(frame_direc)

	name2path = filename2path(source_direc)

	p = mp.Pool()
	for f in os.listdir(vis_direc):
		if f[-3:] == 'png':
			source = name2path[f]
			dest = os.path.join(frame_direc, f)
			print('Copying', source, 'to', dest)
			p.apply_async(copyfile, [source, dest])

	p.close()
	p.join()

