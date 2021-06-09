import numpy as np
import pandas as pd
import timeit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

#function to encode letters into numbers
def encoding(targets):
    encd = []
    for i in range(len(targets)):
        encd.append(ord(targets[i]))
    encd = pd.Series(encd)
    return encd

def main():
    #importing features
    features_data = pd.read_csv(r'C:\Users\dania\Documents\Python\Letter Recognition\letter-recognition_data.csv')
    print('Checking for any missing data \n', features_data.isnull().sum())
    print()
    print('Counting each letter frequency \n', features_data.iloc[:,0].value_counts())

    #adding columns names
    features_data.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 14, 15]

    #Standardizing the dataset (mean = 0 and variance = 1)
    feature_values = features_data.iloc[:, 1:] #passing only features (no letters)
    std_ftrs = StandardScaler().fit_transform(feature_values)

    #dimensionality reduction
    pca = PCA(n_components = 12) #reducing number of features from 16 to 12
    p_components = pca.fit_transform(std_ftrs)
    pca_data = pd.DataFrame(data = p_components)
    
    #Using ASCII values to convert letters (labels) into integers
    labels = features_data.iloc[:, 0]
    encoded_labels = encoding(labels) #using encoding function
    
    #adding labels again to the dataset
    final_ftrs = pd.concat([encoded_labels, pca_data], axis = 1)
    final_ftrs.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] #adding columns names

    #splitting data into train data and test data
    X, y = final_ftrs.iloc[:, 1:], final_ftrs.iloc[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    #applying XGBoost machine learning model
    start_xgb = timeit.default_timer()
    xgb_classifier = xgb.XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=80,
    subsample=0.3, colsample_bytree=0.8)
    xgb_classifier.fit(X_train, y_train)

    #predicting test data
    predictions = xgb_classifier.predict(X_test)
    print('Predictions\n', predictions)
    end_xgb = timeit.default_timer()
    duration_xgb = end_xgb - start_xgb
    print('Run time of XGBoost: {} seconds'.format(round(duration_xgb, 2)))

    #Accuracy measures
    accuracy = (predictions == y_test).sum().astype(float) / len(predictions)*100
    print('Accuracy: ', accuracy)
    start_cv = timeit.default_timer()
    scores = cross_val_score(xgb_classifier, X_test, y_test, cv=3, scoring = "accuracy")
    end_cv = timeit.default_timer()
    duration_cv = end_cv - start_cv
    print('Run time of Cross Validation: {} seconds'.format(round(duration_cv, 2)))
    print('Cross Validation scores: \n', scores)
    precision_xgb = precision_score(y_test, predictions, average='macro')
    print("Precision score: ", precision_xgb)
    recall_xgb = recall_score(y_test, predictions, average='macro')
    print("Recall score: ", recall_xgb)
    f1_xgb =f1_score(y_test, predictions, average = 'macro')
    print('F1 score: ', f1_xgb)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print('Root Mean Squared error: ', rmse)

    #Confusion matrix
    class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R',
    'S','T','U','V','W','X','Y','Z']
    title = 'Letter Recognition Confusion Matrix'
    disp = plot_confusion_matrix(xgb_classifier, X_test, y_test, display_labels=class_names, 
    cmap='Blues')
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    plt.show()

if __name__ == "__main__":main()
