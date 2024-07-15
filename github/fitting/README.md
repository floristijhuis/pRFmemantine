## Snellius pRF fitting procedure
This folder contains the computational architecture that was used to perform pRF fitting of the DN model on the Snellius supercomputing cluster. Concretely, the to-be-fitted vertices were divided into 20 different slices, with each slice sent to one node in the computing cluster for parallellization purposes. 

In the main folder, the `prf_analysis.yml` file contains the settings used for pRF fitting. The `design_task_2R.mat` file is a 100(pixels)x100(pixels)x225(volumes) representation of the design matrix throughout the visual experiment. The `scripts` folder contains the master script used for pRF fitting, given the subject, session, and data slice. The `submissions` folder contains the necessary files to create individual job files for each subject, session, and data slice, to be sent to the SLURM queue. Running the `make_slurm_scripts_correct.py` script leads to population (and execution) of job files in the JOBS folder using `template_2folds.sh` and `examp_yaml.yml` as template files.


