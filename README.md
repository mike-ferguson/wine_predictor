wine_predictor

**About**
Full Stack, End-to-End Machine Learning project that predcits wine success(based on sommelier's rating). 
from certain attributes, like alchol content, acidity, etc. Based off of https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn.
This program encorpaorates data cleaning and preprocessing, hyperparameter tuning, and predcition.

The model is a random forrest regressor, and uses Random Grid Search to find a suitable list of hyperparamters. Then normal 
grid searching to fine-tune a narrow class of hyperparameters based on the random search. 

Tuning can still be done, but as of now, an MSE of 0.380 was acheived, with a accuracy of 92.86% on the test set.

**To Run**:
1) Clone this repo. it contains the main.py program, as well as the datasheet wine-quality.csv
2) Either run main.py in a terminal, or load it into an IDE and run it.
#) The program will run the grid searches and output the predictions.
