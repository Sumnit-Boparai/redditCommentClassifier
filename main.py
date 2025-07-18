#Toxic vs Non-Toxic Comments

#Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split #Used to split the data into test and train
from sklearn.feature_extraction.text import TfidfVectorizer #Used to change the words into number for the machine to understand
from sklearn.naive_bayes import MultinomialNB #Multinomial Niave Bayes Algo which will be used for the classification
from sklearn.metrics import classification_report #Used to neatly print out the stats for the classfication like Precision, Recall, F1, and Support
from sklearn.model_selection import cross_val_score #Yahh Crossvalidation to evaluate the model!

#Load Data
df = pd.read_csv('./train.csv') #Make a table from the CSV with same columns and rows

#Feature Engineering
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000) #Will autolowercase and ignore stopwords in english like "and, the, but" and limit to 10000 more common words
X = vectorizer.fit_transform(df['comment_text']) #Calcualtes the TF-IDF numbers for each word and example
y = df['toxic'] #1 for toxic, 0 for non toxic (y is what we want to predict)

#Training/Testing Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)#Split data into 80% train, 20% test, split using randomseed of 42 so it can be reproduced

#Train Model
clf = MultinomialNB()
clf.fit(X_train, y_train)

#Find out how good it is
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))