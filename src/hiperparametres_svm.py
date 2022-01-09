# -*- coding: utf-8 -*-

from imports import *


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

def hiperparametres_svm(x_train_cv, y_train):
    """
    Búsqueda de l'hiperparàmetre del svm.
    Paràmetres:
        x_train_cv (objecte pandas): part dels atributs de la base de dades per a fer l'entrenament.
        y_train (objecte pandas): part del que es vol predir de la base de dades per a fer l'entrenament.
    Retorna: 
        (dict): diccionari amb la millor C i el millor nucli.
    """
    tuned_parameters = [{'kernel': ['linear', 'poly', 'rbf'],'C': [1]}]
    
    clf = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy')
    clf.fit(x_train_cv, y_train) #trainingdata_without_labels, trainingdata_labels)
    
    return clf.best_params_
