import re

import nltk
import numpy as np
import pandas as pd
from dask_ml.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from toolkit.io_handler import HistoriqueCsv, Import
from toolkit.metrics import macro_disparate_impact
from toolkit.txt_prep import prepareTxt, prepareTxtSpacy

# Création du dico des `noms de métiers`
jobNames = pd.read_csv("./data/categories_string.csv")["0"].to_dict()

# Importation du jeu de données d'entrainement
trainX = Import(fileName="./data/train", fileExtension="json").set_index("Id")

trainY = pd.read_csv("./data/train_label.csv", index_col="Id")

# Concatenation en un seul DataFrame pour visualiser
trainDF = pd.concat([trainX, trainY], axis=1)

# utilisation de la fonction prepareTxt
# trainDF['description']=trainDF.apply(lambda x:prepareTxt(x['description']),axis=1)

# utilisation de la fonction prepareTxt
trainDF["description"] = prepareTxtSpacy(trainDF["description"])


# clf=LinearDiscriminantAnalysis()
# clf=MultinomialNB()
# clf=AdaBoostClassifier(n_estimators=100)
# clf=GradientBoostingClassifier(n_estimators=10)

clf = SGDClassifier(loss="modified_huber", penalty="l2", early_stopping=True)
# clf = RandomForestClassifier(n_jobs=4)

text_clf = Pipeline([("tfidf", TfidfVectorizer()), ("clf", clf),])


trainDF.rename(columns={"Category": "Y"}, inplace=True)
trainSet, validSet = train_test_split(trainDF, shuffle=True, test_size=0.3)


XTrainSet = trainSet.loc[:, trainSet.columns != "Y"]
XValidSet = validSet.loc[:, validSet.columns != "Y"]


YTrainSet = trainSet["Y"]

YValidSet = validSet["Y"]

# Fitting our train data to the pipeline
text_clf.fit(XTrainSet.description, YTrainSet)

predicted = text_clf.predict(XValidSet.description)


Macrof1 = f1_score(predicted, YValidSet, average="macro")
print(f"f1 score =  {Macrof1:.5}")
y_pred = pd.Series(predicted, name="job", index=XValidSet.index)
test_people = pd.concat((y_pred, XValidSet.gender), axis="columns")
fairness = macro_disparate_impact(test_people)
print(f"Fairness = {fairness:.5}")

HistoriqueCsv(
    "PYC",
    "prepareTxtSpacy, TfidfVectorizer()",
    'SGDClassifier(loss="modified_huber", penalty="l2",early_stopping=True)',
    "Hold-out, test_size=0.3",
    Macrof1,
    fairness,
)
