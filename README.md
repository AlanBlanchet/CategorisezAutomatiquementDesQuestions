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
conda create --file environment.yml
conda activate oc
```

# Execution

```bash
jupyter notebook
```

# Matériel

Les scripts de ce repo git ont été lancées avec les informations hardware et logiciels suivants :

```bash

```
