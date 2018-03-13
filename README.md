# DecisionTreeBuilder

Ce programme a été réalisé en binôme avec Jonathan Deramaix dans le cadre du module « Apprentissage artificiel » de notre Master I en Informatique à l'Université d'Angers (2017-2018).


## Génération de règles

Il implémente un algorithme permettant de générer un certain nombre de règles pour prédire la valeur d'un attribut cible dans un tableau de données d'attributs nominaux au format arff.

#### Génération de règles sur le jeu de données weather_nominal :
_python dtbuild.py --file db/weather_nominal.arff_ 



## Mode verbose

Afin d'obtenir plus d'informations sur les données chargées (affichage des tableaux de données de la classe positive et des données de la classe négative, taille des données, dictionnaire des couples attribut-valeur), un mode verbose est disponible.

#### Génération de règles en mode verbose
_python dtbuild.py --file db/weather_nominal.arff -v_


## Choix de l'attribut cible et de sa valeur

Par défaut, l'algorithme considère que le dernier attribut du tableau est l'attribut cible pour lequel il faut générer des règles, et que la première valeur de cet attibut définie dans le fichier arff est la classe positive. Si ce n'est pas le cas, deux arguments optionnels sont disponibles.

#### Recherche des règles pour prédire les cas où le temps incite à ne pas faire de sport (note : ici l'argument --target-col n'est pas nécessaire)
_python dtbuild.py --file db/weather_nominal.arff --target-class no  --target-col play_


## Stratégie pour gérer les valeurs manquantes

L'algorithme va par défaut ignorer les lignes qui contiennent des valeurs manquantes. Une autre stratégie disponible consiste à remplacer les valeurs manquantes par une des valeurs possibles de l'attribut.

#### Stratégie par défaut pour gérer les valeurs manquantes
_python dtbuild.py --file db/vote.arff --missing-values ignore -v_

#### Stratégie remplacant les valeurs manquantes de façon aléatoire
_python dtbuild.py --file db/vote.arff --missing-values randomly_replace -v_


## Mode prédiction

L'algorithme offre également la possibilité de scinder automatiquement le jeu de données en un jeu d'entraînement et un jeu de test, de générer les règles à partir du jeu d'entraînement, et de prédire les valeurs du jeu de test. Il calcule ensuite la précision, le rappel et le score f1 de la prédiction.

#### Prédiction sur le jeu vote.arff
_python dtbuild.py --file db/vote.arff --mode prediction -v_

#### Prédiction sur le jeu vote.arff en utilisant une autre stratégie pour gérer les valeurs manquantes
_python dtbuild.py --file db/vote.arff --mode prediction --missing-values randomly_replace -v_

#### Prédiction sur le jeu mushroom_train.arff
_python dtbuild.py --file db/mushroom_train.arff --mode prediction -v --target-class 1_


