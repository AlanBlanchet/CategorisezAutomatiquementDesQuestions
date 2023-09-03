Projet disponible sur [github](https://github.com/AlanBlanchet/CategorisezAutomatiquementDesQuestions)

# Introduction

- Auteur : Alan Blanchet
- Ecole : OpenClassrooms
- Mentor Ecole : Chemseddine Nabti
- Tuteur entreprise : Arthur Derathé

# Installation

Aller sur [StackExchange](https://data.stackexchange.com/stackoverflow/query/new)

Y insérer cette requête :

```SQL
SELECT TOP 500000 Title, Body, Tags
FROM Posts
WHERE PostTypeId = 1 AND ViewCount > 10 AND Score > 5 AND AnswerCount > 0 AND LEN(Tags) - LEN(REPLACE(Tags, '<','')) >= 5
```

Lancer la requête puis clicker sur "Download CSV"

Placer le CSV dans le dossier "data"

Lancer la commande (nécessite [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html))

```bash
poetry install
```

Il faut ensuite utiliser le kernel créé dans le notebook

# Démo

http://oracle.alan-blanchet.fr:8050/

Voici un répertoire avec du texte à copier/coller si vous êtes en manque d'inspiration

# Matériel

Les scripts de ce repo git ont été lancées avec les informations hardware et logiciels suivants :

```bash
16GiB System memory
Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz
GeForce RTX 2070 Mobile
```
