# Feature-based Prediction of Sound Correspondences / Sound Changes (FPSC)

This repository features two neural models that can predict pairwise sound similarity and the likelihood of sound changes respectively. Since the models operate on phonological features, rather than on a finite alphabet of sounds, they are able to process any pair of sounds represented in IPA.

For both models, a similar sequential workflow was employed, which can be outlined as follows:
1. Large-scale lexical data was aggregated by merging different [Lexibank](https://github.com/lexibank) datasets.
2. Information about the sounds present in the dataset was inferred by the [Etymological Inference Engine](http://sfs.uni-tuebingen.de/~jdellert/talks/jdellert-2019-08-22.pdf) (EtInEn; under development). This information constitutes a first assessment about the likelihood of sounds corresponding to each other, or changing into each other respectively, and was used to generate training data for the neural models.
3. A neural model was trained based on the data generated in the previous step. Sound pairs were represented as a combination of their respective phonological features, enabling the model to generalize learnt patterns over arbitrary input sounds.

For the sound change model, every relevant step is documented in detail in my Master's thesis (Chapter 3, pp 33--47).

## Executable scripts

This repository features 5 executable Python scripts that are all located on the top level directory:

- [eval_sound_transitions.py](eval_sound_transitions.py) -- a throwaway script that calculates sound transition probabilities from the logits, as done in the current EtInEn implementation.
- [generate_extended_params.py](generate_extended_params.py) -- a throwaway script for generating a parameter CLDF file for the merged dataset that includes Concepticon IDs.
- [merge_lexibank_data.py](merge_lexibank_data.py) -- entry script for generating the dataset for the sound change model.
- [train_change_model.py](train_change_model.py) -- script for training the sound change model from transition counts (included in the directory), for more details see below.
- [train_corr_model.py](train_corr_model.py) -- script for training the sound correspondence model from EtInEn-generated PMI scores.

## Training a change model

The current sound change model was trained as a noise-contrastive binary classifier, as decribed in Section 3.2 of my MA thesis. The [train_change_model.py](train_change_model.py) script allows for both, reproducing my model(s), as well as training a classifier on own data.

### Input data

The function `generate_change_train_data` creates training data from a file storing directed transition counts between sounds. The input files used in my workflow are located under `resources/models/change/input`.

The function expects a tab-separated value (`.tsv`) file, containing any sort of transition counts. The first row and the first column are expected to be the alphabet of observed sounds and should therefore be identical. The rest of the file should be a table with numeric values, indicating how often a certain sound changes into another. Hereby, the rows indicate the source sounds, while the columns indicate the target sounds. The row with the symbol "a" therefore contains the information about how often an \[a\] became which other sound.

In order to save memory, the function contains a `compression_rate` parameter that scales the resulting matrix down by the chosen factor. After that, the matrix is rounded to integer values. The input file therefore does not necessarily need to contain integer values, floating-point values (for example, if transitions are weighted) can also be processed.

All information pertaining sounds that can not be represented in terms of phonological features is deleted from the input data. Additionally, a list of symbols that should be disregarded can be specified with the `exclude_symbols` parameter.

### Training a model

The function `train_model` is the backbone for training both the change and the correspondence models. It conveniently wraps the compilation and training of a basic feed-forward neural network using Tensorflow. The only required arguments are `X` and `y` - the input and output data respectively - but a set of keyword arguments can be specified additionally:

- `width`: the width of the hidden layers. *default:* 128.
- `depth`: the depth of the neural network, i.e. the number of hidden layers, including the input layer. *default*: 3.
- `input_dim`: the dimensionality of the input, i.e. the length of the feature vector. *default:* the length of the feature vector.
- `activation`: the activation function to be used for the hidden layers. Needs to be an identifier string supported by Tensorflow. *default:* `gelu`.
- `output_activation`: an activation function to apply to the output layer, if necessary (e.g. to apply the sigmoid function for a binary classifier). *default*: `None`.
- `dropout`: the dropout rate to apply between hidden layers. *default:* 0.05.
- `save_dir`: the directory where the trained model should be saved.
- `model_name`: the name under which the model should be saved.
- `loss`: the loss function to apply for training. *default:* `mean_squared_loss`.
- `test_split`: which fraction of the data should be used for testing the model (and therefore should be excluded from the training data). *default:* 0.1
- `early_stopping`: the patience for early stopping, if desired. *default:* `None`.
- `train_epochs`: for how many epochs the model should be trained. *default:* 200.
- `plot`: whether to produce scatter plots that visualize how well the predicted values converge with the actual target outcomes. *default:* `False`.
- `verbose`: whether to training and test metrics should be printed to the console. *default:* `False`.
