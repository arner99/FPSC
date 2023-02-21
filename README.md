# Feature-based Prediction of Sound Correspondences / Sound Changes (FPSC)

This repository features two neural models that can predict pairwise sound similarity and the likelihood of sound changes respectively. Since the models operate on phonological features, rather than on a finite alphabet of sounds, they are able to process any pair of sounds represented in IPA.

For both models, a similar sequential workflow was employed, which can be outlined as follows:
1. Large-scale lexical data was aggregated by merging different [Lexibank](https://github.com/lexibank) datasets.
2. Information about the sounds present in the dataset was inferred by the [Etymological Inference Engine](http://sfs.uni-tuebingen.de/~jdellert/talks/jdellert-2019-08-22.pdf) (EtInEn; under development). This information constitutes a first assessment about the likelihood of sounds corresponding to each other, or changing into each other respectively, and was used to generate training data for the neural models.
3. A neural model was trained based on the data generated in the previous step. Sound pairs were represented as a combination of their respective phonological features, enabling the model to generalize learnt patterns over arbitrary input sounds.

For the sound change model, every relevant step is documented in detail in my Master's thesis (Chapter 3, pp 33--47).

## Executable scripts

This repository features 5 executable Python scripts that are all located on the top level directory:

- <eval_sound_transitions.py> -- a throwaway script that calculates sound transition probabilities from the logits, as done in the current EtInEn implementation.
- <generate_extended_params.py> -- a throwaway script for generating a parameter CLDF file for the merged dataset that includes Concepticon IDs.
- <merge_lexibank_data.py> -- entry script for generating the dataset for the sound change model.
- <train_change_model.py> -- script for training the sound change model from transition counts (included in the directory).
- <train_corr_model.py> -- script for training the sound correspondence model from EtInEn-generated PMI scores.
