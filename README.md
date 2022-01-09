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

## Demo

## Conclusions

## Idees per a treballar en un futur





Classificació de diferents enllaços webs a diferents categories. 
