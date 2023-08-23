
# MLOps Project
### Description

Vous êtes bénévole pour l'association de protection des animaux, Agir pour les animaux, de votre quartier. Vous vous demandez donc ce que vous pouvez faire en retour pour aider l'association.
Vous apprenez, en discutant avec un bénévole, que leur base de données de pensionnaires commence à s'agrandir et qu'ils n'ont pas toujours le temps de référencer les images des animaux qu'ils ont accumulées depuis plusieurs années. Ils aimeraient donc obtenir un algorithme capable de classer les images en fonction de la race du chien présent sur l'image.

### Mission
L'association vous demande de réaliser un algorithme de détection de la race du chien sur une photo, afin d'accélérer leur travail d’indexation. Il s'agit également de mettre en place un dashboard permettant de tester le modèle choisi et suivre les bonnes pratiques MLOps pour ce projet.



## Classification automatique de la race des chiens

**Données utilisées**


[Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/): une base de données d'environ 20000 images de chiens réparties sur 120 races.

Le jeu de données est subdivisé en deux parties il y'a un dossier `images`et un dossier `annotation`. Les deux renseignent les mêmes informations. Dans images par exemple, vous trouverez des dossiers et chque dossier correspond à une race et contient plusieurs images. C'est un jeu de données assez fourni et toutes les classes sont bien renseignées. 

**Prétraitement des données**

Pour le nettoyage des données, il a été effectuer un redimensionnement, une égalisation et un débruitage avec le filtre Non-local Means Filter.

Le dataset de départ a été réduit pour se retrouver avec 2500 images pour des soucis de performance. Ainsi une augmentation des données s'est imposée lors de l'entrainement des modèles


**Modeling**
Plusieurs modèles ont été testé:
- **CNN from scratch**: nous allons utiliser une architecture simple et peu profonde qui nous servira de point de départ pour les modèles futurs.
    Cependant ce dernier n'a pas donné des      résutats satisfaisant. En effet ne donne pas de bons résultats. En effet,l'accuracy ne dépasse pas 25% sur nos données

- **Xception transfert learning**: une architecture de réseau de neurones convolutionnels (CNN) qui a été pré-entraînée sur de vastes jeux de données, tels que ImageNet.
    Le modèle Xception est dérivé de l'architecture Inception. Inception a pour but de réduire la consommation de ressources des CNN profonds.
    Ce modèle a donné des résutats assez satisafaisant sur nos données avec une accuracy de plus de 80%.

- **Restnet**: pour ne pas s’arrêter là, un autre transfert learninga été fait sur le modèle Restnet qui a donné également de très bons résutats derrière Xception, avec une accuray de près de 80%.

    Ainsi le modèle Xception présente de meilleurs résultats. On a donc continué avec ce dernier afin de l'optimiser pour de meilleures performances. 
    `Optuna` et `kerastuner` ont été utilié pour l'optimisation des hyperparamètres du modèle.

**Tracking du best model**:  MLFlow.

![MLflow Screenshot](/screenshots/mlflow1.png?raw=true)



**Prédictions du meilleur modèle**
![Model Screenshot](/screenshots/predictions.png?raw=true)


**Application**

Afin d'utiliser le modèle, une application web a été développé pour faciliser la classification des chiens en race.

![App Screenshot](/screenshots/app2.jpg?raw=true)


![App Screenshot](/screenshots/app1.jpg?raw=true)




## Structure du projet
```bash
.
├── README.md
├── mlruns
│   ├── 0
│   │   └── meta.yaml
│   └── 708235932076291406
│       ├── 067ce031f32a4b02b0fd68762e0915e7
│       │   ├── meta.yaml
│       │   ├── metrics
│       │   │   ├── final_train_accuracy
│       │   │   └── final_train_loss
│       │   └── tags
│       │       ├── mlflow.log-model.history
│       │       ├── mlflow.note.content
│       │       ├── mlflow.runName
│       │       ├── mlflow.source.name
│       │       ├── mlflow.source.type
│       │       ├── mlflow.user
│       │       ├── priority.pbm
│       │       └── version
│       ├── 9bc9619f14d64f0fba1abf315dbe644c
│       │   ├── meta.yaml
│       │   ├── metrics
│       │   │   ├── final_train_accuracy
│       │   │   └── final_train_loss
│       │   └── tags
│       │       ├── mlflow.note.content
│       │       ├── mlflow.runName
│       │       ├── mlflow.source.name
│       │       ├── mlflow.source.type
│       │       ├── mlflow.user
│       │       ├── priority.pbm
│       │       └── version
│       ├── c8a3773ae53345b48cacebe792a2a858
│       │   ├── meta.yaml
│       │   ├── metrics
│       │   │   ├── final_train_accuracy
│       │   │   └── final_train_loss
│       │   └── tags
│       │       ├── mlflow.log-model.history
│       │       ├── mlflow.note.content
│       │       ├── mlflow.runName
│       │       ├── mlflow.source.name
│       │       ├── mlflow.source.type
│       │       ├── mlflow.user
│       │       ├── priority.pbm
│       │       └── version
│       ├── c8c016e9165a4c20b2f8a1d0461d5767
│       │   ├── artifacts
│       │   │   └── best_xception_model
│       │   │       ├── MLmodel
│       │   │       ├── conda.yaml
│       │   │       ├── python_env.yaml
│       │   │       └── requirements.txt
│       │   ├── meta.yaml
│       │   └── tags
│       │       ├── mlflow.log-model.history
│       │       ├── mlflow.note.content
│       │       ├── mlflow.runName
│       │       ├── mlflow.source.name
│       │       ├── mlflow.source.type
│       │       ├── mlflow.user
│       │       ├── priority.pbm
│       │       └── version
│       ├── d1aabb96a9de4af193231b18d9f95ad2
│       │   ├── meta.yaml
│       │   └── tags
│       │       ├── mlflow.note.content
│       │       ├── mlflow.runName
│       │       ├── mlflow.source.name
│       │       ├── mlflow.source.type
│       │       ├── mlflow.user
│       │       ├── priority.pbm
│       │       └── version
│       ├── e6e6bfb6e091451aa4a86a3bb64c845b
│       │   ├── meta.yaml
│       │   └── tags
│       │       ├── mlflow.note.content
│       │       ├── mlflow.runName
│       │       ├── mlflow.source.name
│       │       ├── mlflow.source.type
│       │       ├── mlflow.user
│       │       ├── priority.pbm
│       │       └── version
│       ├── f4bcd52e3bde40e0bae64476b18f17cf
│       │   ├── meta.yaml
│       │   └── tags
│       │       ├── mlflow.note.content
│       │       ├── mlflow.runName
│       │       ├── mlflow.source.name
│       │       ├── mlflow.source.type
│       │       ├── mlflow.user
│       │       ├── priority.pbm
│       │       └── version
│       └── meta.yaml
├── models
│   ├── 20230821_model_stanford_breed_dogs.h5
│   ├── 20230822_model_stanford_breed_dogs.h5
│   └── encoder_classes.npy
├── notebooks
│   └── mlops_stanford_breed_dogs_model.ipynb
├── requirements.txt
├── screenshots
│   └── mlflow1.png
└── settings
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-310(1).pyc
    │   ├── __init__.cpython-310.pyc
    │   ├── params.cpython-310(1).pyc
    │   └── params.cpython-310.pyc
    └── params.py
```