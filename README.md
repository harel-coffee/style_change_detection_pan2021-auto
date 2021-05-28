# PAN 2021 Style Change Detection

This repository contains the source code used to generate our model for the PAN 2021 style change detection shared task: https://pan.webis.de/clef21/pan21-web/style-change-detection.html

We have included the trained models that were used in my submission. However, the training datasets are not included in the repository and can be downloaded from: https://pan.webis.de/data.html


## Reproducing Results
1. Download the datasets and place them in `data/pan2021/` directory. 
2. Preprocessing: To preprocess the data, run `preprocess_data.ipynb`. This schould create the preprocessed files in `preprocessed_data/2021/` direcotry. Preprocessed data are not included in this repo
4. Model training: Run `train_model.ipynb`. These scripts read the preprocessed files, extracts features, and trains the model. The feature vectorizers and the finals models are included in the repository. 
5. Predicting and measuring performance: Run `predict.ipynb`. We used the PAN 2021 evaluator script to compute all the performance metrics (Script obtained from: https://github.com/pan-webis-de/pan-code/blob/master/clef21/style-change-detection/evaluator.py).
5. Predicting: `predict.py` is the files used to make predictions using the trained model on TIRA.

Please contact janith@nyu.edu if you have trouble running these scripts.  