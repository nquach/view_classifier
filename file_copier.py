import os
from shutil import copyfile
import multiprocessing as mp
from utils import *

if __name__ == '__main__':
	root = '/scratch/groups/willhies/echo_view_classifier/'

	source_direc = '/scratch/groups/willhies/echo_view_classifier/frames'
	A4C_source = os.path.join(source_direc, 'A4C_frames')
	PLAX_source = os.path.join(source_direc, 'PLAX_frames')
	PSAX_source = os.path.join(source_direc, 'PSAX_frames')

	vis_direc = os.path.join(root, 'guided_backprop_vis')
	A4C_vis_direc = os.path.join(vis_direc, 'A4C')
	PLAX_vis_direc = os.path.join(vis_direc, 'PLAX')
	PSAX_vis_direc = os.path.join(vis_direc, 'PSAX')

	frame_direc = os.path.join(vis_direc, 'greyscale')
	A4C_grey = os.path.join(frame_direc, 'A4C')
	PLAX_grey = os.path.join(frame_direc, 'PLAX')
	PSAX_grey = os.path.join(frame_direc, 'PSAX')

	safe_makedir(A4C_grey)
	safe_makedir(PLAX_grey)
	safe_makedir(PSAX_grey)

	A4C_filenames = os.listdir(A4C_vis_direc)
	PLAX_filenames = os.listdir(PLAX_vis_direc)
	PSAX_filenames = os.listdir(PSAX_vis_direc)

	p = mp.Pool()
	for f in A4C_filenames:
		source = os.path.join(A4C_source, f)
		dest = os.path.join(A4C_grey, f)
		print('Copying', source, 'to', dest)
		p.apply_async(copyfile, [source, dest])

	for f in PLAX_filenames:
		source = os.path.join(PLAX_source, f)
		dest = os.path.join(PLAX_grey, f)
		print('Copying', source, 'to', dest)
		p.apply_async(copyfile, [source, dest])

	for f in PSAX_filenames:
		source = os.path.join(PSAX_source, f)
		dest = os.path.join(PSAX_grey, f)
		print('Copying', source, 'to', dest)
		p.apply_async(copyfile, [source, dest])

	p.close()
	p.join()

