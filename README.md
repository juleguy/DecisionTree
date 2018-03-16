# DecisionTreeBuilder

Ce programme a été réalisé en binôme avec Jonathan Deramaix dans le cadre du module « Apprentissage artificiel » de notre Master I en Informatique à l'Université d'Angers (2017-2018).

## Dépendances
Ce programme nécessite les librairies suivantes : pandas, arff2pandas, sklearn et scipy. Elles sont installables en utilisant la commande pip. Pour pouvoir exécuter le programme sans installer les librairies, on peut utiliser l'environnement virtuel python inclus :

#### Utilisation de l'environnement virtuel python :
```shell
./venv/bin/python3.5 dtbuild.py --arguments
```

## Génération de règles

Ce programme implémente un algorithme permettant de générer un certain nombre de règles pour prédire la valeur d'un attribut cible dans un tableau de données d'attributs nominaux au format arff.

#### Génération de règles sur le jeu de données weather_nominal :
```shell
python dtbuild.py --file db/weather_nominal.arff
```

## Mode verbose

Afin d'obtenir plus d'informations sur les données chargées (affichage des tableaux de données de la classe positive et des données de la classe négative, taille des données, dictionnaire des couples attribut-valeur), un mode verbose est disponible.

#### Génération de règles en mode verbose :
```shell
python dtbuild.py --file db/weather_nominal.arff -v
```


## Choix de l'attribut cible et de sa valeur

Par défaut, l'algorithme considère que le dernier attribut du tableau est l'attribut cible pour lequel il faut générer des règles, et que la première valeur de cet attibut définie dans le fichier arff est la classe positive. Si ce n'est pas le cas, deux arguments optionnels sont disponibles.

#### Recherche des règles pour prédire les cas où le temps incite à ne pas faire de sport (note : ici l'argument --target-col n'est pas nécessaire) :
```shell
python dtbuild.py --file db/weather_nominal.arff --target-class no  --target-col play
```

## Mode prédiction

L'algorithme offre également la possibilité de scinder automatiquement le jeu de données en un jeu d'entraînement et un jeu de test, de générer les règles à partir du jeu d'entraînement, et de prédire les valeurs du jeu de test. Il calcule ensuite la précision, le rappel et le score f1 de la prédiction.

#### Prédiction sur le jeu vote.arff :
```shell
python dtbuild.py --file db/vote.arff --mode prediction -v
```

#### Prédiction sur le jeu mushroom_train.arff (traitement plus long) :
```shell
python dtbuild.py --file db/mushroom_train.arff --mode prediction -v --target-class 1
```


## Stratégie pour gérer les valeurs manquantes

L'algorithme va par défaut ignorer les lignes qui contiennent des valeurs manquantes. Une autre stratégie disponible consiste à remplacer les valeurs manquantes par une des valeurs possibles de l'attribut. Une dernière stratégie remplace les valeurs manquantes par la valeur la plus présente pour l'attribut dans la classe courante (positive ou négative) et dans le jeu courant (jeu d'entraînement ou jeu de test).

#### Stratégie par défaut pour gérer les valeurs manquantes, en mode prédiction :
```shell
python dtbuild.py --file db/vote.arff --missing-values ignore --mode prediction -v
```

#### Stratégie remplaçant les valeurs manquantes de façon aléatoire, en mode prédiction :
```shell
python dtbuild.py --file db/vote.arff --missing-values randomly_replace --mode prediction -v
```

#### Stratégie remplaçant les valeurs manquantes par la valeur de l'attribut possédant le plus d'occurences pour la classe de l'exemple, en mode prédiction :
```shell
python dtbuild.py --file db/vote.arff --missing-values most_common_replace --mode prediction -v
```


