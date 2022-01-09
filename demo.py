# -*- coding: utf-8 -*-

# Marina Vazquez Guerrero (NIU: 1563735)

# arxius que s'han d'importar que contenen les funcions que utilitzem
from generate_features import *
from train_models import *
from score_models import *
from hiperparametres import *

data = carrega_data("website_classification.csv") # carreguem la base de dades cridant a la funcio corresponent
data,cv = neteja_data(data, ["Category"], ["cleaned_website_text"], ["website_url", "Unnamed: 0"]) # netejem la base de dades

# dividim la base de dades en un 60% train i la resta de test
X_train,X_test,y_train,y_test = train_test_split(data["cleaned_website_text"], data["Category"], train_size = 0.6)
# adaptem la part x per a que sigui útil ja que principalment són descripcions de les diferents webs
x_train_cv = cv.transform(X_train)
x_test_cv = cv.transform(X_test)
x_cv = cv.transform(data["cleaned_website_text"])

# entrenaments i tests dels millors models 
train_regressio_logistica(x_train_cv, y_train, "1", guardar_model = True, C = 20309.17620904739, t = 0.001)
acc_rl, tm_rl = score_models_simples("1lr.sav", x_test_cv, y_test)
print("Model: Regressió Logística         |  Precisió:",round(acc_rl,16)," |  Temps:",tm_rl,"segons")

c = hiperparametres_svm(x_train_cv, y_train)
train_svm(x_train_cv, y_train, nom_output = "1", guardar_model = True, k = c["kernel"], C = c["C"])
acc_svm, tm_svm = score_models_simples("1svm.sav", x_test_cv, y_test)
print("Model: SVM (k =",c["kernel"], "i  C =", c["C"],")  |  Precisió:",round(acc_svm,16)," |  Temps:",tm_svm,"segons")

train_random_forest(x_train_cv, y_train, nom_output = "1", guardar_model = True, i = 220)
acc_rf, tm_rf = score_models_simples("1rf.sav", x_test_cv, y_test)
print("Model:Random Forest (220 arbres)   |  Precisió:",round(acc_rf,16)," |  Temps:",tm_rf,"segons")

train_bagg_svm(x_train_cv, y_train, nom_output = "1", guardar_model = True, k = c["kernel"], C = c["C"])
acc_bknn, tm_bknn = score_models_simples("1BaggSVM.sav", x_test_cv, y_test)
print("Model: Bagging SVM                 |  Precisió:",round(acc_bknn,16)," |  Temps:",tm_bknn,"segons")

train_bagg_rl(x_train_cv, y_train, nom_output = "1", guardar_model = True, C = 20309.17620904739, t = 0.001)
acc_brl, tm_brl = score_models_simples("1BaggRL.sav", x_test_cv, y_test)
print("Model: Bagging Regressió Logística |  Precisió:",round(acc_brl,16)," |  Temps:",tm_brl,"segons")
