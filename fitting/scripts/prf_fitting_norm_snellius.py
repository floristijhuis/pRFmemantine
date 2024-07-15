import yaml
import os
from prfpy import stimulus, model, fit
import numpy as np
from scipy import io
import nibabel as nib
from datetime import datetime, timedelta
import time
import sys

opj = os.path.join

# define constants
eps = 1e-1
inf = np.inf

########################################################################################
# set parameters
########################################################################################

basedir = '/home/ftijhuis'

subindex = str(sys.argv[1])
njobs = int(sys.argv[2])
slice_n = int(sys.argv[3])
ses = int(sys.argv[4])

nbatches=njobs

sub=f'sub-{subindex}'
ses=f'ses-{ses}'
deriv=opj(basedir, 'data')

with open(opj(basedir, 'prf_analysis.yml')) as f:
    analysis_info = yaml.safe_load(f)

dm = io.loadmat(opj(basedir, 'design_task-2R.mat'))['stim']
dm = dm[:,:,5:]

# define the pRF stimulus
prf_stim = stimulus.PRFStimulus2D(
    screen_size_cm=analysis_info['screen_size_cm'],
    screen_distance_cm=analysis_info['screen_distance_cm'],
    design_matrix=dm, 
    TR=analysis_info['TR'])

# define settings based on prf stimulus:
ss = prf_stim.screen_size_degrees
max_ecc_size = ss/2.0

#rsq thresh
rsq_threshold = analysis_info['rsq_threshold']

# define the grids. this is primarily relevant for gaussian model, as the outputs from the gaussian models are put as grid into norm model
grid_nr = 30
sizes, eccs, polars = max_ecc_size * np.linspace(0.25, 1, grid_nr)**2, \
    max_ecc_size * np.linspace(0.1, 1, grid_nr)**2, \
    np.linspace(0, 2*np.pi, grid_nr)

# these are for the norm model
surround_amplitude_grid = analysis_info['norm']['surround_amplitude_grid']
surround_size_grid = analysis_info['norm']['surround_size_grid']
neural_baseline_grid = analysis_info['norm']['neural_baseline_grid']
surround_baseline_grid = analysis_info['norm']['surround_baseline_grid']

# this is for the fitting
xtol = analysis_info['xtol']
ftol = analysis_info['ftol']

#define the HRF
hrf_pars = analysis_info['hrf']['pars'] # SHOULD THIS BE 1, 1, 0 (JURJEN) OR 1, 4.6, 0 (MARCO?)

# define the bounds for your model
gauss_bounds = [(-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
                (-1.5*max_ecc_size, 1.5*max_ecc_size),  # y
                (0.2, 1.5*ss),  # prf size
                tuple(analysis_info['prf_ampl_gauss']),  # prf amplitude
                tuple(analysis_info['bold_bsl']), # bold baseline SHOULD THIS BE 0 OR 1000?
                tuple(analysis_info['hrf']['deriv_bound']), # hrf derivative
                tuple(analysis_info['hrf']['disp_bound'])]  # hrf dispersion

norm_bounds = [(-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
                (-1.5*max_ecc_size, 1.5*max_ecc_size),  # y
                (0.2, 1.5*ss),  # prf size
                tuple(analysis_info['prf_ampl_norm']),  # prf amplitude
                tuple(analysis_info['bold_bsl']),  # bold baseline SHOULD THIS BE 0 OR 1000?
                tuple(analysis_info['norm']['surround_amplitude_bound']),  # surround amplitude
                (eps, 3*ss),  # surround size
                tuple(analysis_info['norm']['neural_baseline_bound']),  # neural baseline
                tuple([float(item) for item in analysis_info['norm']['surround_baseline_bound']]), # surround baseline
                tuple(analysis_info['hrf']['deriv_bound']), # hrf derivative
                tuple(analysis_info['hrf']['disp_bound'])]  # hrf dispersion

print(f'Gauss bounds are: {gauss_bounds}')
print(f'Norm bounds are: {norm_bounds}')

print(f'Gauss grids are: {sizes}, {eccs}, {polars}')
print(f'Norm grids are: {surround_amplitude_grid}, {surround_size_grid}, {neural_baseline_grid}, {surround_baseline_grid}')

########################################################################################
# load the data
########################################################################################

data_full = np.load(opj(deriv, sub, 'data', ses, f'{sub}_{ses}_task-2R_hemi-LR_desc-avg_bold.npy'))
  
#Load V1 vertices from left hemisphere, together with the number of vertices in the left hemisphere so indexing for right hemisphere works appropriately; the .label files start from 0 and work per hemisphere, whereas Inkscape merges the two hemispheres together.
V1_lh = nib.freesurfer.read_label(opj(deriv, sub, 'rois', 'lh.V1.label'))
V2_lh = nib.freesurfer.read_label(opj(deriv, sub, 'rois', 'lh.V2.label'))
V3_lh = nib.freesurfer.read_label(opj(deriv, sub, 'rois', 'lh.V3.label'))

all_lh = nib.freesurfer.read_geometry(opj(deriv, sub, 'rois', 'surf', 'lh.inflated'))

#Load V1 vertices from right hemisphere. 
V1_rh = nib.freesurfer.read_label(opj(deriv, sub, 'rois', 'rh.V1.label'))
V2_rh = nib.freesurfer.read_label(opj(deriv, sub, 'rois', 'rh.V2.label'))
V3_rh = nib.freesurfer.read_label(opj(deriv, sub, 'rois', 'rh.V3.label'))

all_rh = nib.freesurfer.read_geometry(opj(deriv, sub, 'rois', 'surf', 'rh.inflated'))

# Get the right indices for the right hemisphere by adding the number of vertices in left hemisphere to the ROI indices for the right hemisphere.
V1_rh = V1_rh + len(all_lh[0])
V2_rh = V2_rh + len(all_lh[0])
V3_rh = V3_rh + len(all_lh[0])

#Concatenate the 2 and sort them such that they are in the right order
V1_vertices = np.sort(np.concatenate([V1_lh, V1_rh]))
V2_vertices = np.sort(np.concatenate([V2_lh, V2_rh]))
V3_vertices = np.sort(np.concatenate([V3_lh, V3_rh]))

roi_vertices = np.sort(np.concatenate([V1_vertices, V2_vertices, V3_vertices]))

# Filter data based on ROIS
vertices_to_fit = np.array_split(roi_vertices, 20)[slice_n] # CHANGE IF YOU WANT TO CUT IT IN EVEN SMALLER SLICES OR BIGGER, MAYBE INCLUDE IN ANALYSIS FILE
np.save(opj(basedir, 'output', sub, 'roivertices', f'roivertices_{sub}_slice-{slice_n}'), vertices_to_fit)

data_roi = data_full[:, vertices_to_fit]

print(f"Fitting data of shape {data_roi.shape}", flush=True)
print("",flush=True)

########################################################################################
# Gaussian fit
########################################################################################

#define model now
gauss_model = model.Iso2DGaussianModel(stimulus=prf_stim, hrf=hrf_pars)
gauss_fitter = fit.Iso2DGaussianFitter(data=data_roi.T, model=gauss_model, n_jobs=njobs, fit_hrf=True)

# first, gridfit
print(f"Now starting Gaussian gridfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", flush=True)
start = time.time()

gauss_fitter.grid_fit(ecc_grid=eccs,
                 polar_grid=polars,
                 size_grid=sizes, 
                 n_batches=nbatches,
                 fixed_grid_baseline=0,
                 grid_bounds=[tuple(analysis_info['prf_ampl_gauss'])],
                 verbose = True)

elapsed = (time.time() - start)

gauss_grid = np.nan_to_num(gauss_fitter.gridsearch_params)
mean_rsq = np.mean(gauss_grid[gauss_grid[:, -1]>rsq_threshold, -1])
        
# verbose stuff
start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
nr = np.sum(gauss_grid[:, -1]>rsq_threshold)
total = gauss_fitter.data.shape[0]
print(f"Completed Gaussian gridfit at {start_time}. Voxels/vertices above {rsq_threshold}: {nr}/{total}",flush=True)
print(f"Gridfit took {timedelta(seconds=elapsed)} | Mean rsq>{rsq_threshold}: {round(mean_rsq,2)}",flush=True)

np.save(opj(basedir, 'output', sub, 'fits', f'{sub}_{ses}_task-2R_model-gauss_stage-grid_desc-prfparams_slice-{slice_n}.npy'), gauss_fitter.gridsearch_params)

# now, iterative fit
print(f"Now starting Gaussian iterfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}",flush=True)
start = time.time()

gauss_fitter.iterative_fit(
    rsq_threshold=rsq_threshold, 
    bounds=gauss_bounds,
    constraints=[],
    xtol=xtol,
    ftol=ftol)

# print summary
elapsed = (time.time() - start)              
gauss_iter = np.nan_to_num(gauss_fitter.iterative_search_params)

# verbose stuff
mean_rsq = np.nanmean(gauss_iter[gauss_fitter.rsq_mask, -1])
start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')

print(f"Completed Gaussian iterfit at {start_time}. Mean rsq>{rsq_threshold}: {round(mean_rsq,2)}",flush=True)
print(f"Iterfit took {timedelta(seconds=elapsed)}",flush=True)

# save parameters
np.save(opj(basedir, 'output', sub, 'fits', f'{sub}_{ses}_task-2R_model-gauss_stage-iter_desc-prfparams_slice-{slice_n}.npy'), gauss_fitter.iterative_search_params)

# BEFORE CONTINUING, THROW OUT THE PRFs WITH A NEGATIVE AMPLITUDE
gauss_fitter.iterative_search_params[gauss_fitter.iterative_search_params[:, 3] < 0] = np.float64(0)

########################################################################################
# DN model fit
########################################################################################

norm_model = model.Norm_Iso2DGaussianModel(stimulus=prf_stim, hrf=hrf_pars)
norm_fitter = fit.Norm_Iso2DGaussianFitter(data=data_roi.T, model=norm_model, previous_gaussian_fitter=gauss_fitter, n_jobs=njobs, fit_hrf=True)

# first, gridfit
print(f"Now starting DN gridfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}",flush=True)
start = time.time()

norm_fitter.grid_fit(surround_amplitude_grid=surround_amplitude_grid,
                surround_size_grid=surround_size_grid,
                neural_baseline_grid=neural_baseline_grid,
                surround_baseline_grid=surround_baseline_grid, gaussian_params=gauss_fitter.iterative_search_params,
                n_batches=nbatches,
                rsq_threshold=rsq_threshold,
                fixed_grid_baseline=0,
                grid_bounds=[tuple(analysis_info['prf_ampl_norm']), tuple(analysis_info['norm']['neural_baseline_bound'])],
                verbose=True)

elapsed = (time.time() - start)

norm_grid = np.nan_to_num(norm_fitter.gridsearch_params)
mean_rsq = np.mean(norm_grid[norm_grid[:, -1]>rsq_threshold, -1])
        
# verbose stuff
start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
nr = np.sum(norm_grid[:, -1]>rsq_threshold)
total = norm_fitter.data.shape[0]
print(f"Completed DN gridfit at {start_time}. Voxels/vertices above {rsq_threshold}: {nr}/{total}",flush=True)
print(f"Gridfit took {timedelta(seconds=elapsed)} | Mean rsq>{rsq_threshold}: {round(mean_rsq,2)}",flush=True)

np.save(opj(basedir, 'output', sub, 'fits', f'{sub}_{ses}_task-2R_model-norm_stage-grid_desc-prfparams_slice-{slice_n}.npy'), norm_fitter.gridsearch_params)

# now, iterative fit
print(f"Now starting DN iterfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}",flush=True)
start = time.time()

norm_fitter.iterative_fit(
    rsq_threshold=rsq_threshold, 
    bounds=norm_bounds,
    constraints=[],
    xtol=xtol,
    ftol=ftol)

# print summary
elapsed = (time.time() - start)              
norm_iter = np.nan_to_num(norm_fitter.iterative_search_params)

# verbose stuff
mean_rsq = np.nanmean(norm_iter[norm_fitter.rsq_mask, -1])
start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')

print(f"Completed DN iterfit at {start_time}. Mean rsq>{rsq_threshold}: {round(mean_rsq,2)}",flush=True)
print(f"Iterfit took {timedelta(seconds=elapsed)}", flush=True)

# save parameters
np.save(opj(basedir, 'output', sub, 'fits', f'{sub}_{ses}_task-2R_model-norm_stage-iter_desc-prfparams_slice-{slice_n}.npy'), norm_fitter.iterative_search_params)
