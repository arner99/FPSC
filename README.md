# Feature-based Prediction of Sound Correspondences (FPSC)

This repository contains all relevant code and data in order to finally obtain a model that can predict PMI scores for a sound pair based on the phonological features of the sounds. 
The workflow can be subdivided in four steps:

1. Extending the panphon database in order to represent diphthongs, triphthongs, and complex tones. To be able to represent those segments, additional features have been introduced.
2. Merging several lexibank databases to obtain stable PMI scores for a large set of sounds
3. Inferring a global correspondence model for the sounds in the newly crafted lexibank dataset, using Information-Weighted Sequence Alignment as described by Dellert (2019).
4. Combining the obtained PMI scores with feature encodings to train a neural network to predict PMI scores for a sound pair.
