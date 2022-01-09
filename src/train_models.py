# -*- coding: utf-8 -*-

# Marina Vazquez Guerrero (NIU: 1563735)

# impotem les llibreries i els paquets que necessitem
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


def train_regressio_logistica(x_t, y_t, nom_output = False, guardar_model = False, C = 2.0, t = 0.001):
    """
    Entrenament del model de Regressio Logistica.
        Paràmetres:
            x_t (objecte pandas): part dels atributs de la base de dades per a fer l'entrenament.
            y_t (objecte pandas): part del que es vol predir de la base de dades per a fer l'entrenament.
            nom_output (string): nom que te el fitxer de l'entrenament del model en cas que ho guardem.
            guardar_model (boolea): boolea per a controlar si volem guardar l'entrenament.
            C (int): hiperparàmetre necessari per al model.
            t (int): tolerancia utilitzada al model.
    """
    logireg = LogisticRegression(C=C, fit_intercept=True, penalty='l2', tol=t, max_iter = 1000)
    logireg.fit(x_t, y_t)

    if (guardar_model):    
        pickle.dump(logireg, open('../models/'+str(nom_output)+'lr.sav', 'wb'))        
def train_svm(x_t, y_t, nom_output = False, guardar_model = False, k = 'linear', C = 10.0):
    """
    Entrenament del model de SVM.
        Paràmetres:
            x_t (objecte pandas): part dels atributs de la base de dades per a fer l'entrenament.
            y_t (objecte pandas): part del que es vol predir de la base de dades per a fer l'entrenament.
            nom_output (string): nom que te el fitxer de l'entrenament del model en cas que ho guardem.
            guardar_model (boolea): boolea per a controlar si volem guardar l'entrenament.
            k (string): kernel utiltzat en l'entrenament del model.
            C (int): hiperparàmetre necessari per al model.
    """
    svc = svm.SVC(C=C, kernel=k,  probability=True)
    svc.fit(x_t, y_t)

    if (guardar_model):    
        pickle.dump(svc, open('../models/'+str(nom_output)+'svm.sav', 'wb'))
        
def train_decision_tree(x_t, y_t, nom_output, guardar_model = False):
    """
    Entrenament d'un arbre de decisio.
        Paràmetres:
            x_t (objecte pandas): part dels atributs de la base de dades per a fer l'entrenament.
            y_t (objecte pandas): part del que es vol predir de la base de dades per a fer l'entrenament.
            nom_output (string): nom que te el fitxer de l'entrenament del model en cas que ho guardem.
            guardar_model (boolea): boolea per a controlar si volem guardar l'entrenament.
    """
    dt = tree.DecisionTreeClassifier()
    dt.fit(x_t, y_t)

    if (guardar_model):    
        pickle.dump(dt, open('../models/'+str(nom_output)+'dt.sav', 'wb'))
        
def train_knn(x_t, y_t, nom_output, guardar_model = False, n = 7):
    """
    Entrenament del KNN.
        Paràmetres:
            x_t (objecte pandas): part dels atributs de la base de dades per a fer l'entrenament.
            y_t (objecte pandas): part del que es vol predir de la base de dades per a fer l'entrenament.
            nom_output (string): nom que te el fitxer de l'entrenament del model en cas que ho guardem.
            guardar_model (boolea): boolea per a controlar si volem guardar l'entrenament.
            n (int): nombre de veins utilitzats en el model.
    """
    knn = neighbors.KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_t,y_t)

    if (guardar_model):    
        pickle.dump(knn, open('../models/'+str(nom_output)+'knn.sav', 'wb'))
        
def train_random_forest(x_t, y_t, nom_output, guardar_model = False, i = 10):
    """
    Entrenament del Random Forest.
        Paràmetres:
            x_t (objecte pandas): part dels atributs de la base de dades per a fer l'entrenament.
            y_t (objecte pandas): part del que es vol predir de la base de dades per a fer l'entrenament.
            nom_output (string): nom que te el fitxer de l'entrenament del model en cas que ho guardem.
            guardar_model (boolea): boolea per a controlar si volem guardar l'entrenament.
            i (int): nombre d'arbres creats en el model.
    """
    rf = RandomForestClassifier(n_estimators=i)
    rf.fit(x_t,y_t)
    
    if (guardar_model):    
        pickle.dump(rf, open('../models/'+str(nom_output)+'rf.sav', 'wb'))
        
def train_kmeans(x_t, x_v, y_t, y_v, nom_output, guardar_model = False, n = 16):  
    """
    Entrenament del KMeans.
        Paràmetres:
            x_t (objecte pandas): part dels atributs de la base de dades per a fer l'entrenament.
            y_t (objecte pandas): part del que es vol predir de la base de dades per a fer l'entrenament.
            nom_output (string): nom que te el fitxer de l'entrenament del model en cas que ho guardem.
            guardar_model (boolea): boolea per a controlar si volem guardar l'entrenament.
            n (int): nombre de classes del model.
    """
    kmeans = KMeans(n_clusters= n)
    kmeans.fit(x_t,y_t)
    
    if (guardar_model):    
        pickle.dump(kmeans, open('../models/'+str(nom_output)+'kmeans.sav', 'wb'))
        
def train_bagg_rl(x_t, y_t, nom_output, guardar_model = False, C = 2.0, t = 0.001):
    """
    Entrenament del Bagging de Regressio Logistica.
        Paràmetres:
            x_t (objecte pandas): part dels atributs de la base de dades per a fer l'entrenament.
            y_t (objecte pandas): part del que es vol predir de la base de dades per a fer l'entrenament.
            nom_output (string): nom que te el fitxer de l'entrenament del model en cas que ho guardem.
            guardar_model (boolea): boolea per a controlar si volem guardar l'entrenament.
            C (int): hiperparàmetre necessari per al model.
            t (int): tolerancia utilitzada al model.
    """
    clf = BaggingClassifier(base_estimator=LogisticRegression(C = C, fit_intercept = True, penalty = 'l2', tol = t), n_estimators=10, random_state=0)
    clf.fit(x_t, y_t)
    
    if (guardar_model):    
        pickle.dump(clf, open('../models/'+str(nom_output)+'BaggRL.sav', 'wb'))
        
def train_bagg_knn(x_t, y_t, nom_output, guardar_model = False, n = 7):
    """
    Entrenament del Bagging de KNN.
        Paràmetres:
            x_t (objecte pandas): part dels atributs de la base de dades per a fer l'entrenament.
            y_t (objecte pandas): part del que es vol predir de la base de dades per a fer l'entrenament.
            nom_output (string): nom que te el fitxer de l'entrenament del model en cas que ho guardem.
            guardar_model (boolea): boolea per a controlar si volem guardar l'entrenament.
            n (int): nombre de veins utilitzats en el model.
    """
    clf = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=n), n_estimators=10, random_state=0)
    clf.fit(x_t, y_t)
    
    if (guardar_model):    
        pickle.dump(clf, open('../models/'+str(nom_output)+'BaggKNN.sav', 'wb'))
    
def train_bagg_svm(x_t, y_t, nom_output, guardar_model = False, k = 'linear', C = 10.0):
    """
    Entrenament del Bagging del SVM.
        Paràmetres:
            x_t (objecte pandas): part dels atributs de la base de dades per a fer l'entrenament.
            y_t (objecte pandas): part del que es vol predir de la base de dades per a fer l'entrenament.
            nom_output (string): nom que te el fitxer de l'entrenament del model en cas que ho guardem.
            guardar_model (boolea): boolea per a controlar si volem guardar l'entrenament.
            k (string): kernel utiltzat en l'entrenament del model.
            C (int): hiperparàmetre necessari per al model.
    """
    clf = BaggingClassifier(base_estimator=svm.SVC(C = C, kernel = k, probability = True), n_estimators=10, random_state=0)
    clf.fit(x_t, y_t)
    
    if (guardar_model):    
        pickle.dump(clf, open('../models/'+str(nom_output)+'BaggSVM.sav', 'wb'))
    
def train_boosting(x_t, x_v, y_t, y_v, nom_output, guardar_model = False, n = 7):
    """
    Entrenament de l'AdaBoost.
        Paràmetres:
            x_t (objecte pandas): part dels atributs de la base de dades per a fer l'entrenament.
            y_t (objecte pandas): part del que es vol predir de la base de dades per a fer l'entrenament.
            nom_output (string): nom que te el fitxer de l'entrenament del model en cas que ho guardem.
            guardar_model (boolea): boolea per a controlar si volem guardar l'entrenament.
            n (int): hiperparàmetre necessari per al model.
    """
    clf = AdaBoostClassifier(n_estimators = n)
    clf.fit(x_t, y_t)
    
    if (guardar_model):    
        pickle.dump(clf, open('../models/'+str(nom_output)+'AdaBoost.sav', 'wb'))
        