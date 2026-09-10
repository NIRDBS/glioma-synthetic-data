Glioma Tumor Grading

Slide 0
Title, author, profesor, logos, date, faculty

Slide 1
Problem in clinical context:
Brain Tumors have low survival rates.
Tumor grading takes days or weeks for histopathology tests.
Raman Spectroscopy could provide results in minutes.

Slide 2
Outline

Slide 3-4
Data structure as a tensor, with 3 axes: 2 spatial and 1 spectral
graphics to show the structure

Slide 5
Data processing
- stack outline: median filter, NMF, GMM, airPLS, normalization, etc. Take the stack from the thesis

Slide 6
median filter

Slide 7
NMF
- explain endmember decomposition, why non-negative is suitable here: light scattering is an additive process, molecule mixture is an additive process. Maps well on the problem, even if we can't guarantee that the factorization will map individual component mixtures. 

Slide 8
GMM
- Math definition
- explain that the NMF coefficients are used as input to the GMM

Slide 9
- clustering and tumor (or relevant) spectra selection

Slide 10-12 (or 10-13 or 10-14?)
- CVAE definition
- explain architecture (encoder, decoder)
- 1D CNN operator
- seen as a function, that is a composition of functions that act linearly (layers) and non linearly (activation fct)
- composite loss function (reconstruction, classification, KL)
- training: Adam optimizer (sgd based plus momentu, adaptive learning, brief mention, could be only oral)
- any other relevant details

Slide 13
CV design

Slide 14
Bootstrap design
- explain why CV at patient level (used in classifiers) and bootstrap at slide level (simulate a dataset of samples)

Slide 15
Results

Slide 16
Discussion
pca, tsne plots interpretation

Slide 17
Conclusion, future work, importance of work