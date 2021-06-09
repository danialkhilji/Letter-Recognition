import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

#loading features
letters_df = pd.read_csv(r'...letter-recognition_data.csv')
print(letters_df.shape)
#Converting letters into integers
letters_df.columns =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 14, 15]
encode = LabelEncoder()
letters_num = pd.Series(encode.fit_transform(letters_df.iloc[:, 0]))
#Scaling features
scaled_features = pd.DataFrame(StandardScaler().fit_transform(letters_df.iloc[:, 1:]))
#Reducing features
pca = PCA(n_components = 8) #reducing number of features from 16 to 12
components = pca.fit_transform(scaled_features)
reduced_features = pd.DataFrame(data = components)
#adding labels again to the dataset
letters_df1 = pd.concat([letters_num, reduced_features], axis = 1)
letters_df1.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#Spliting dataset
data, labels = letters_df1.iloc[:, 1:], letters_df1.iloc[:,0]
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.25)
#KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(train_data, train_labels)
predictions = knn_model.predict(test_data)
print('Predictions: ', predictions)
print("Accuracy: ", metrics.accuracy_score(test_labels, predictions))
cross_scores = cross_val_score(knn_model, test_data, test_labels, cv=3, scoring='accuracy')
print('Scores: \n', cross_scores)
#Confusion matrix
letters_label = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R',
'S','T','U','V','W','X','Y','Z']
disp = plot_confusion_matrix(knn_model, test_data, test_labels, display_labels=letters_label, 
cmap='plasma')
disp.ax_.set_title('Confusion Matrix')
print('Confusion Matrix')
print(disp.confusion_matrix)
plt.show()