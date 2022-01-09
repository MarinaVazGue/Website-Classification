# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score,accuracy_score
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm, tree, datasets, neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import completeness_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


def score_models_simples(file, x_v, y_v):
    """
    Precisió dels entrenaments dels diferents models simples.
    Paràmetres:
        file (string): nom del fitxer de l'entrenament que es vol utilitzar per a calcular la precisió del model.
        x_v (objecte pandas): part dels atributs de la base de dades per a fer el test.
        y_v (objecte pandas): part del que es vol predir de la base de dades per a fer el test.
    Retorna:
        (float): precisió del model.
        (float): temps que es triga en fer les prediccions.
    """
    clf = pickle.load(open(file, 'rb'))
    i = time.time()
    predictions = clf.predict(x_v)
    f = time.time()
    return accuracy_score(y_v,predictions), f-i

def score_kmeans(file, x_v, y_v):
    """
    Precisió dels entrenaments del model Kmeans.
    Paràmetres:
        file (string): nom del fitxer de l'entrenament que es vol utilitzar per a calcular la precisió del model.
        x_v (objecte pandas): part dels atributs de la base de dades per a fer el test.
        y_v (objecte pandas): part del que es vol predir de la base de dades per a fer el test.
    Retorna:
        (float): precisió del model.
        (float): temps que es triga en fer les prediccions.
    """
    kmeans = pickle.load(open(file, 'rb'))
    i = time.time()
    predictions = kmeans.predict(x_v)
    f = time.time()
    return completeness_score(y_v,predictions), f-i
    
def score_boosting_bagging(file, x_v, y_v):
    """
    Precisió dels entrenaments de tipus Bagging i Boosting.
    Paràmetres:
        file (string): nom del fitxer de l'entrenament que es vol utilitzar per a calcular la precisió del model.
        x_v (objecte pandas): part dels atributs de la base de dades per a fer el test.
        y_v (objecte pandas): part del que es vol predir de la base de dades per a fer el test.
    Retorna:
        (float): precisió del model.
        (float): temps que es triga en fer les prediccions.
    """
    bg = pickle.load(open(file, 'rb'))
    i = time.time()
    y_pred = bg.predict(x_v)
    f = time.time()
    return accuracy_score(y_v, y_pred), f-i
