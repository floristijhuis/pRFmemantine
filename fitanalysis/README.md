## Analysis of pRF model fits 
This folder contains the scripts used to statistically analyze the output of model fits of the memantine project.

- `dnmodel_widget.ipynb` is an interactive widget to explore the effects of different DN model parameters on the pRF profile shape.
- `scanquality.ipynb` is a notebook that explores within-subject and between-condition differences in tSNR and FD.
- `averagedata_gaussianfit_analysis.ipynb` is a notebook that was used to calculate ROI sizes and fit quality within ROIs based on the Gaussian fit on the average data from the two sessions.
- `gauss_vs_norm_fitanalysis.ipynb` is a notebook that compares the Gaussian and subsequent DN model fits for all vertices in V1-V3.
- `plac_vs_mem_fitanalysis.ipynb` is a notebook that compares the DN model fits between the placebo and memantine condition for all vertices in V1-V3.
- `surround_effects.ipynb` is the notebook containing the primary analysis of the project; the comparison of the effect of memantine on pRF characteristics of V1 vertices. 
- `surround_effects_prfsize.ipynb` is a notebook that is similar to the first part of `memantine_effects.ipynb`. However, it contains a post-hoc analysis in which vertices are binned based on pRF size rather than eccentricity. 

