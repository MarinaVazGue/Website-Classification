# -*- coding: utf-8 -*- 
from train_models import *

# Marina Vazquez Guerrero (NIU: 1563735)


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def carrega_data(csv):
    """
    Carrega la base de dades a partir d'un arxiu csv.
        Paràmetres:
            csv (string): arxiu que conté la base de dades en un csv.
        Retorna: 
            (objecte pandas): base de dades en pandas. 
    """
    return pd.read_csv(csv,encoding='utf-8') #pd.read_csv(csv)
    
def neteja_data(data, lst_encoder = [], lst_cv = [], lst_drop = []):
    """
    Neteja del codi a partir de la informacio passada amb els parametres. Primer de tot, 
    si hi ha algun atribut que s'ha de categoritzar amb el LabelEncoder, es fa. Despres es 
    mira si a algun atribut s'ha d'aplicar el CountVectorizer i, per ultim, els que s'han 
    d'eliminar. Finalment, es comprova si hi ha algun NaN en la base de dades i, si n'hi ha,
    es posa un 0.
        Paràmetres:
            data (objecte pandas): base de dades en pandas.
            lst_encoder (list): llista amb els atributs que s'han d'aplicar el LabelEncoder.
            lst_cv (list): llista amb els atributs que s'han d'aplicar el CountVectorizer.
            lst_drop (list): llista amb els atributs que es volen eliminar.
        Retorna: 
            data (objecte pandas): base de dades en pandas neta. 
    """
    for e in lst_encoder:
        le = LabelEncoder()
        data[e] = le.fit(data[e]).transform(data[e])
    
    for c in lst_cv:
        data[c] = data[c].astype(str)
        cv = CountVectorizer(analyzer = "word", stop_words = "english")
        cv.fit(data[c])
            
    for d in lst_drop:
        data.drop([d], axis=1, inplace=True)
        
    for at in data:
        if data.isnull().sum()[at] != 0:
            data[at] = data[at].fillna(0)
            
    return data,cv