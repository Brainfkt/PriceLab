# PriceLab - Pricing Intelligence Simulator

## Phrase objectif final

PriceLab est un simulateur d'intelligence pricing qui transforme l'historique des ventes, des prix, des stocks, des promotions et du contexte marché en recommandations de fourchettes de prix fiables par produit, avec estimation de l'impact sur volume, chiffre d'affaires, marge, risque et niveau de confiance.

## Prompt de départ pour planifier le projet

```text
Tu es à la fois Product Manager senior, Data Scientist senior, ML Engineer et UX architect. Je veux planifier un projet portfolio Data Scientist appelé PriceLab.

OBJECTIF DU PROJET
Créer une application locale d'intelligence pricing pour un catalogue de produits. L'application doit analyser l'historique d'achats, l'évolution des prix, les volumes vendus, les stocks, les promotions et plusieurs variables contextuelles afin de :
1. comprendre quels prix vendent le mieux ;
2. estimer l'élasticité prix de chaque produit ;
3. détecter les périodes où les prix fonctionnent mieux ;
4. simuler des scénarios de prix ;
5. recommander des fourchettes de prix selon l'objectif : volume, chiffre d'affaires, marge ou prudence ;
6. afficher un score de fiabilité pour éviter les recommandations abusives ;
7. produire des explications lisibles pour un recruteur, un analyste pricing ou un décideur business.

CONTEXTE
Ce projet est destiné à un portfolio de Data Scientist. Il doit montrer une expertise sérieuse : data cleaning, feature engineering, statistique, machine learning, backtesting temporel, interprétabilité, scoring de fiabilité, simulation et UX analytique. Je ne veux pas un simple dashboard descriptif. Je veux un outil qui ressemble à un produit ML exploitable.

STACK RECOMMANDEE POUR LE MVP
- Python 3.11+
- Streamlit pour l'interface
- pandas, numpy, scipy
- scikit-learn pour les modèles
- plotly pour les visualisations
- pydantic pour la validation du schéma
- pytest pour les tests
- joblib pour sauvegarder les modèles si utile
- aucun service payant obligatoire
- dataset synthétique généré automatiquement au premier lancement

DONNEES ATTENDUES
L'application doit fonctionner avec un CSV importé par l'utilisateur et avec un dataset démo généré automatiquement. Le grain recommandé est product_id x date x channel x region, avec agrégation quotidienne ou hebdomadaire.

Colonnes minimales :
- date
- product_id
- product_name
- category
- units_sold
- price
- cost
- stock_available
- promotion_flag

Colonnes optionnelles utiles :
- discount_rate
- channel
- region
- competitor_price
- marketing_spend
- traffic
- holiday_flag
- customer_segment
- returns
- weather_index

MODULES FONCTIONNELS A PLANIFIER
1. Data import et schema mapper.
2. Data quality scanner.
3. Catalogue overview.
4. Product deep dive.
5. Price performance matrix.
6. Elasticity engine.
7. Best moment detector.
8. Promotion analyzer.
9. Scenario simulator.
10. Optimal price finder.
11. Catalogue opportunity scanner.
12. Model explainability panel.
13. Reliability score.
14. Export d'un rapport produit en Markdown ou HTML.

REGLES DE FIABILITE IMPORTANTES
L'application doit refuser ou dégrader les recommandations quand :
- l'historique est trop court ;
- le produit a trop peu de variations de prix ;
- le prix est quasiment constant ;
- les ruptures de stock sont trop fréquentes ;
- les promotions contaminent trop l'historique ;
- le modèle prédit mal sur backtesting temporel ;
- le scénario demandé extrapole trop loin des prix observés ;
- les coûts sont absents alors que l'objectif est la marge.

ATTENDU DE TA REPONSE
Avant toute ligne de code, produis un plan très détaillé comprenant :
1. la vision produit ;
2. les hypothèses retenues ;
3. le périmètre MVP et le hors-périmètre ;
4. l'architecture technique ;
5. l'arborescence complète du repo ;
6. le schéma de données ;
7. le pipeline de traitement ;
8. les features à créer ;
9. les métriques business et data science ;
10. les modèles à utiliser et pourquoi ;
11. la méthode de backtesting ;
12. la formule du reliability score ;
13. les écrans de l'application ;
14. les tests unitaires et tests d'acceptation ;
15. les risques statistiques et les garde-fous ;
16. la roadmap en étapes implémentables ;
17. les critères qui permettent de dire que le projet est terminé.

CONTRAINTES DE REPONSE
- Ne commence pas à coder.
- Ne propose pas une architecture inutilement complexe.
- Priorise un MVP robuste, local, démontrable et lisible par un recruteur.
- Si une information manque, pose au maximum 5 questions, puis propose des hypothèses par défaut.
- Fais des choix concrets au lieu de rester vague.
- Prévois des commandes de lancement, de test et de validation.
```

## Vision produit

PriceLab n'est pas un simple dashboard de ventes. C'est un outil d'aide à la décision qui analyse comment le prix, le temps, la promotion, le stock et le contexte influencent la demande. Le projet doit donner l'impression d'un produit ML sérieux : il mesure son incertitude, explique ses limites et bloque les recommandations fragiles.

## Modules essentiels

1. Data import et schema mapper.
2. Data quality scanner.
3. Catalogue overview.
4. Product deep dive.
5. Price performance matrix.
6. Elasticity engine.
7. Optimal price finder.
8. Best moment detector.
9. Promotion analyzer.
10. Scenario simulator.
11. Catalogue opportunity scanner.
12. Model explainability panel.
13. Reliability score.
14. Export rapport produit.

## Stack MVP

Python 3.11+, Streamlit, pandas, numpy, scipy, scikit-learn, plotly, pydantic, pytest, joblib.

## Schéma minimal

- date
- product_id
- product_name
- category
- units_sold
- price
- cost
- stock_available
- promotion_flag

## Prompt de réalisation après planification

```text
A partir du plan validé, implémente le MVP complet de PriceLab dans un nouveau repository Python.

Contraintes :
- application Streamlit locale ;
- dataset synthétique généré automatiquement si aucun CSV n'est fourni ;
- import CSV avec mapping de colonnes ;
- modules séparés dans src/ ;
- tests avec pytest ;
- README complet ;
- code typé autant que possible ;
- aucun service externe obligatoire ;
- visualisations Plotly ;
- modèle interprétable + modèle ML prédictif ;
- backtesting temporel ;
- reliability score ;
- recommandations de prix avec garde-fous.

Livrables attendus :
1. tous les fichiers du projet ;
2. commandes d'installation et de lancement ;
3. tests exécutables ;
4. dataset démo réaliste ;
5. explication des choix de modélisation ;
6. liste des limites statistiques ;
7. checklist finale de validation.

Ne termine pas tant que l'application ne se lance pas localement et que les tests de base ne passent pas.
```
