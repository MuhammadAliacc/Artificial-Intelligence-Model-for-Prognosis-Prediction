# Artificial-Intelligence-Model-for-Prognosis-Prediction
This project presents an in-depth analysis of prognosis data using advanced machine learning techniques. The primary goal of this project is to develop a robust neural network model to predict patient outcomes based on various prognostic factors. The project leverages the power of Python, popular deep learning libraries, and the Optuna library for hyperparameter optimization.

# Motivation
Accurate prognosis is crucial in healthcare to guide medical decision-making and improve patient outcomes. The motivation behind this project is to develop a predictive model that can aid healthcare professionals in making informed decisions for their patients. By harnessing the potential of neural networks and optimizing their performance with Optuna, we aim to create a reliable tool for prognosis prediction.

# Data set used 
The dataset used in this project is sourced from Kaggle and contains a diverse set of prognostic features for a large number of patients. It includes demographic information, medical history, laboratory results, and other relevant factors that influence patient prognosis. 

# Data preprocessing (Feacher Engineering)
Before fitting the model, we conducted a series of essential data preprocessing steps to ensure the best performance of our neural network. These steps include:
* Feature Combination: We combined relevant features to create more informative and higher-level features, enhancing the model's ability to capture complex relationships.
* Feature Clustering: Employing unsupervised learning techniques, we clustered similar features to reduce dimensionality and enhance the interpretability of the model.
* Feature Selection: Using various selection methods such as recursive feature elimination and statistical tests, we identified the most important features that significantly contribute to the prediction task.

# K-Fold Cross-Validation:
To robustly evaluate the performance of our neural network model, we adopted the k-fold cross-validation method. The dataset was divided into k subsets (folds), and we trained and evaluated the model k times, each time using a different fold as the validation set and the remaining k-1 folds as the training set. This approach helps us to obtain more reliable performance metrics and minimize overfitting.

# Usage
To reproduce the results or apply the model to other prognosis datasets, clone the GitHub repository and follow the instructions provided in the README file. You can customize the neural network architecture and experiment with different hyperparameters using Optuna to further improve the model's performance.

#
### Libraries Used
* [Optuna](https://optuna.org/) : For hyperparameter tuning
* [NumPy](https://numpy.org/) : Fundamental package for scientific computing
* [pandas](https://pandas.pydata.org/) : Used for manipulation and analysis of dataframes
* [scikit-learn](https://scikit-learn.org/stable/) : Library used to implement machine learning, and related methods
* [TensorFlow](https://www.tensorflow.org/) : Used for AI based models and methods

# 
### Contributors to the Source Code
* [Yuganshu Wadhwa](https://github.com/YuganshuWadhwa) 
* [Maximilian Brandt](https://github.com/brandeyy) 
* [Muhammad Ali](https://github.com/MuhammadAliacc) 
