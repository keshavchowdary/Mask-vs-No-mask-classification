# Mask-vs-No-mask-classification
An end to end ML model to classify whether a person is wearing a mask or not
I.  ABSTRACT
This project is based on the classification of the masked and unmasked images. The aim of our project is to model three classifiers, the reasons behind choosing these classifiers and compare the performance efficiency of all these classifiers.
II. INTRODUCTION
The project we worked on is to classify whether a person is wearing a mask or not and for this we have used a self-built-masked-face-recognition-dataset. This dataset contains two sets of images, one set contains the images of persons with masks and the other set contains the images without masks. In each set there are sub directories with the name of person and in each of these sub directories there are images.
III. MACHINE LEARNING PIPELINE
Preprocessing
The first step in the machine learning pipeline is to preprocess the data. Import the basic dependencies such as pandas, numpy and matplotlib. Here we have images and convert these images in such a way that the system understands it. So we have converted the images into 64*64 pixel values for each R,G and B. By using the pillow library in python we have converted the images into pixel values corresponding to RGB. So in total we have 12,288 columns. Now we have the data in the shape (64,64,3), so we have converted it into a single row and created a dataframe separately for both masked and unmasked data. Then add a target variable (0 for unmasked and 1 for masked) to the masked and unmasked dataframes. Then concatenate these two dataframes into a single one and shuffle it to avoid any kind of bias.
Dimensionality reduction
The next step in the machine learning pipeline is dimensionality reduction/feature selection. We have done dimensionality reduction. First of all, What is Dimensionality reduction and why do we do it? . In the given dataset, we have a total of 12,288 input features, so handling 12,288 columns is time consuming. So to overcome that we reduce the dimensions using various techniques like PCA, LDA. Here we have done dimensionality reduction using Principal component analysis (PCA).  PCA basically projects this multi dimensional data into a new feature space with less no of dimensions and the data is variated along the principal component axes.
PCA in detail:
Standardization of the data i.e. making the mean 0 and variance 1 for all features to avoid bias towards high feature values.
Compute the covariance matrix.
Calculate the eigenvalues and eigenvectors.
Determine the feature vector.
Transform the data along the principal component axes.
Model training and evaluation
The next step in the machine learning pipeline is to train the data with a suitable model.
Models used are,
RandomForest Classifier
RandomForest Classifier is a supervised algorithm that fits the data using decision trees as the base estimators. This classifier makes decision trees using some random subsets in the dataset and collects the data from all the decision trees by voting and makes the final prediction.
GaussianNB Classifier:
This classifier assumes that all the features are independent of each other and considers all of these properties to independently contribute to the probability that the image belongs to a particular class and the probability is calculated using bayes theorem.
  
P(y) is the prior probability, P(X/y) is the likelihood and P(X) is the evidence. It calculates the posterior probability for a sample being classified into different target classes and the class with highest probability will be the target class corresponding to that sample.
K-Nearest Neighbors:
This is a supervised classification algorithm that works by calculating the distance between the test data and each row in the training dataset (in general euclidean distance is considered) and sort all those distance in ascending order and chooses the k nearest points and assign the most frequent class to the test data.
We used naive bayes because the calculation of probabilities for each sample are simplified to make the calculation traceable.
Randomforest is selected because the decision tree (weak learner) performs better on the binary classification.
The KNN algorithm is one of the best classifiers for the binary classification because the knn works based on the similarity of the features and here we have only one check point to consider. So this model works better.
Evaluating the parameters for each model
For each model I have evaluated the model with different parameters using GridSearchCV and the best_parameters obtained are:
For RandomForest model:
{'max_depth': 6, 'max_features': 'sqrt', 'min_samples_leaf': 8}
For GaussianNB model:
{'var_smoothing':0.08111308307896872}
For KNN model:
{'n_neighbors': 7, 'p': 5}

Then trained the model with corresponding best_params_ which are obtained from GridSearchCV.
