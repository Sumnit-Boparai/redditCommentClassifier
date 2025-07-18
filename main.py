#Toxic vs Non-Toxic Comments

#Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer #Used to change the words into number for the machine to understand
from sklearn.naive_bayes import MultinomialNB #Multinomial Niave Bayes Algo which will be used for the classification
from sklearn.metrics import classification_report #Used to neatly print out the stats for the classfication like Precision, Recall, F1, and Support
from sklearn.model_selection import cross_val_predict #Yahh Crossvalidation to evaluate the model!

#Load Data
df = pd.read_csv('./train.csv') #Make a table from the CSV with same columns and rows

#Feature Engineering
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000) #Will autolowercase and ignore stopwords in english like "and, the, but" and limit to 10000 more common words
X = vectorizer.fit_transform(df['comment_text']) #Calcualtes the TF-IDF numbers for each word and example
y = df['toxic'] #1 for toxic, 0 for non toxic (y is what we want to predict)

#Train Model
clf = MultinomialNB()
y_pred = cross_val_predict(clf, X, y, cv = 10)

#Find out how good it is
print(classification_report(y, y_pred))