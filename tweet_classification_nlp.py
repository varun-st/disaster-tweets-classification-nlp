from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import ensemble
import pandas as pd 
import xgboost
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils



def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)
  
# reading csv file  
df = pd.read_csv("2013_Brazil_nightclub_fire-tweets_labeled.csv") 

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['Tweet Text'], df['Informativeness'])


# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
dummy_y = np_utils.to_categorical(train_y)
dummyv_y = np_utils.to_categorical(valid_y)

#print(list(encoder.classes_))

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(df['Tweet Text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

f = open("count_result.txt","w")


accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
f.write("Multinomial Naive Bayes: " +  str(round(accuracy * 100,2)) + "%\n")

accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
f.write("Logistic Regression: " + str(round(accuracy * 100,2)) + "%\n")

accuracy = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)
f.write("Support Vector Matix: " + str(round(accuracy * 100,2)) + "%\n")

accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
f.write("Random Forrest: " +  str(round(accuracy * 100,2)) + "%\n")

accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
f.write("Extreme Gradient Boosting: " + str(round(accuracy * 100,2)) + "%\n")

# neural network
model = Sequential()
model.add(Dense(2248,input_dim=3369, activation='sigmoid'))
#model.add(Dense(100, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xtrain_count, dummy_y, epochs=100, batch_size=5,verbose=0)
scores = model.evaluate(xvalid_count, dummyv_y,verbose=0)
f.write("Neural Network: " + str(round(scores[1]*100,2)) + "%\n")

f.close()





