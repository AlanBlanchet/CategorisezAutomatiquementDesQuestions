# Explication

Ce projet ayant pour but de faire prédire notre IA plusieurs tags pour nos questions stack overflow, il faut donc une approche multi-label.

Je vais donc utiliser mes connaissances et métriques passées pour faire cette approche multi-label.

Le gros changement est que nous n'allons pas utiliser uniquement `target1` mais également `2,3,4,5`

## BERT

J'ai donc effectué des tests en multi-label sur le modèle BERT.

Ce que j'en conclu c'est qu'il est plus facile de converger vers des résultats cohérent avec les modèles classiques sklearn comme le LogisticRegression.

En effet, il semblait que BERT avait tendance à trop prédire au lieux de favoriser la qualité des prédictions par rapport aux ground truth.
