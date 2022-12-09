# Group 1

Our idea was to use multiple feature extraction models, combine them as a vector, and put it through a FCNN. For some feature_models we separate pure ECG data into templates and inverted the polarity of the ECG signal if needed. So the schema looks like

Input → flatten(ensemble(feature_model_0, …, feature_model_n)) → FCNN → output

We trained all feature_models as a single model together with FCNN and saved the weights for the trained model. Then we put the models together and left the weights frozen (this performed the best). Each feature_model also requires its own data loader. To help models not overfit we are also augmenting the features (when applicable).

Feature models can be described as:

# Group 2

1. 1D Inception net. The input of the network is the direct ECG data (we cut the samples for each batch → inputs have to be the same length for each batch). It contains 6 (layers of) inception blocks.
2. 1D Inception net with transformers. The input is 1D templates of ECG. Network also adds transformers encoder after inception blocks.
3. 1D Inception net (another). Network is the same as 1. but the input is RRI instead of direct ECG.
4. 2D Convnet (best performer). The input to the model is (2d) Mel spectrogram. The input matrix is then fed through 8 conv (which increase in size), batchnorm, and pooling layers.
5. 2D residual convnet. Very to the previous one but uses 4 2d residual blocks + pooling.

FCNN has 5 fully connected layers (also trained with dropout).

The were two main parts to this task: 1. Feature Extraction and 2. Classification

1. Feature Extraction
We used many different libraries to extract as many useful features as we could from the raw signal data given to us. The most relevant libraries we used were neurokit2, catch22, TSFEL, emd and tqdm. We also crafted plenty of features ourselves. We also trained a convolutional network on the spectrograms of the ecg signals. We implemented a ResNet-type architecture with residual skip connections after every three convolutional layers. In total we used 9 convolutional layers, pooled the result and fed it through a 2-layer fully connected classifier. The results were unfortunately as good as we had hoped, but we could nonetheless extract the features in the penultimate layer and used them further down the line for our classification model. After having collected all our features, of which we had around 800, we removed those that were highly correlated. This removed quite a few, since as we were using many different libraries we  produced some duplicates. 

2. Classification
Once we had our many (not highly correlated) features, we decided to fit a LGBM and XGBoost classifiers to them. After some hyperparameter tuning we got scores above 0.85, and then finally to reach our submission scored we took a majority vote over all the predictions we had collected so far.

# Group 3

Manual FE:
We use neurokit2 and biospy2 to extract the P, Q, R, S, T peaks and compute their relative positions, their amplitudes, their intervals and their relations. We computed some statistical measures on them as well.
We use the libraries hrv-analysis and heartpy to get the heart rate variability and other non-linear features. 
We also extract some fft features.

We remove constant features, fill nan values with 0 and "SelectKBest" 240 features.


We split the data into 5 folds and create a ML1 and ML2 model for every fold. 

ML1 FE:
We use a resnet model based on "ENCASE: an ENsemble ClASsifiEr for ECG Classification Using Expert Features and Deep Neural Networks";. The model has 48 convolutional blocks, where each block consists of two 1d convolutional, batchnorm, maxpool and drop out layers. We extract 32 features form the last layer.

ML2 FE:
We implemented the model described in "Towards Understanding ECG Rhythm Classification Using Convolutional Neural Networks and Attention Mappings". We extract the 64 features from the last layer. 

Loss function: Cross Entropy (no balanced weights)
Preprocessing Data: use neurokit clean, resample signal to 6000 for ML1, zeropad to 18000 for ML2


Final Estimator: StackingClassifier with LGBM, XGBoost, RandomForest and HistGradient.
Final Prediction: Hard Classification of 5 estimators, where each estimator is trained on different ML features from different folds.

Direkt zu:
Laden Sie die mobile App
Datenschutzinfos
