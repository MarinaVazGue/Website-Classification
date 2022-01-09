# Pràctica Kaggle APC UAB 2021-22
### Nom: Marina Vázquez Guerrero
### DATASET: Website-Classification
### URL: [Kaggle](https://www.kaggle.com/hetulmehta/website-classification)

## Resum
El dataset utilitzat s'ha extret de la web Kaggle. Conté 1408 mostres amb 4 atributs dels quals 1 és de tipus numèric enter que descriu l'índex de la mostra, el segon té les URL, el tercer és un atribut amb llenguatge natural i l'últim és de tipus categòric ja que categoritza les diferents webs en 16 tipus diferents.

### Objectius del dataset
Volem saber la categoria d'una web a partir del seu URL i d'una petita descripció. 

### Experiments
Durant aquesta pràctica hem realitzat diferents experiments a partir dels diferents tipus de models utilitzats. Hem fet servir la Regressió Logística, també models d'aprenentange tant supervisats com no supervisats. A més, hem fet Bagging i Boosting. Amb tots aquests experiments, hem pogut observar que els que tenen millor rendiment amb aquesta base de dades són els d'aprenentagte supervisat i la Regressió logística.

### Preprocessat
El preprocessament d'aquesta base de dades no ha estat molt costosa. En primer lloc, amb la visualització de les primeres files de la base de dades és observable que tant l'atribut *Unnamed: 0* com *website_url* no són importants i que es podrien eliminar. D'aquesta manera, ens quedem amb l'atribut *cleaned_website_text* i el que volem predir, *Category*. En segon lloc, s'ha de tenir en compte de quin tipus és cada atribut per a saber si podem treballar amb ells directament o cal fer-ne alguna cosa abans. Veiem que *cleaned_website_text* és una descripció de les webs que està en llenguatge natural i que, com no és de tipus numèric, hem d'utilitzar el *CountVectorizer* el qual converteix el text en una matriu de recomptes de símbols. Seguidament, a l'atribut *Category* hem d'aplicar el _LabelEncoder_ ja que volem que es categoritzi per números i no amb _strings_ com teniem fins ara. Finalment, mirem si hi ha algun _NaN_ en la base de dades i veiem que no hi ha cap i que, per tant, hem acabat el preprocessament perquè no tenim cap atribut que haguem de normalitzar.

### Model
Els resultats de les diferents proves dels models són les següents:

| Model | <div style="width:220px">Hiperparàmetres</div> | Mètrica| Temps |
|-- | -- | -- | -- |
| **Regressió Logístoca** | C = 2.0, tol = 0.001 | 94.15% | 0.00795s |
| **SVM** | C = 1, k = linear | 91.49%  |  1.09609s |
| **Arbre de decisió** | - | 82.092% | 0.00199s |
| **KNN** | K = 1 | 84.75% | 0.09782s |
| **KNN** | K = 7 | 50.00% | 0.09196s |
| **Random Forest** | Arbres = 220 | 93.26% | 0.26717s |
| **Kmeans** | K = 25 | 0.002% | 0.02619s |
| **Bagging SVM** | C = 1, k = linear | 90.07% | 8.28453s |
| **Bagging Regresiió Logística** | C = 2.0, tol = 0.001 | 94.8% | 0.25331s |
| **Bagging KNN** | K = 1 | 83.98% | 2.11115s |
| **Bagging KNN** | K = 7 | 48.40% | 0.72146s |
| **Bagging Ada Boost** | - | 16.134% | 0.07828s |

## Demo
Per a poder executar-ho i veure els diferents resultats, tenim el fitxer _demo.py_ on hi ha una mostra dels entrenaments i tests dels models amb millors resultats. Per tal de fer una prova, es pot fer servir amb la següent comanda:
``` python demo.py ```

## Conclusions
El millor model que s'ha aconseguit ha estat la Regressió Logística i, seguidament, el SVM amb nucli lineal. Això es pot veure d'una manera molt ràpida amb la taula anterior de resultats dels models. És destacable veure que fent el Bagging d'aquests dos models, que són amb els que obtenim una precisió més alta, no obtenim resultats més significants, sinó que surten de molt similars.

## Idees per a treballar en un futur
Aquesta pràctica podria ser millorable fent servir models més complexos. També, es podria indagar en l'ús de models de Pytorch ja que té més opcions i utilitats en general. 
