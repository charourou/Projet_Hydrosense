# 💧 Hydro-Sense : Prédire l'Invisible

> Anticiper les seuils d'alerte des nappes phréatiques grâce au Machine Learning.

Projet réalisé dans le cadre du bootcamp **Le Wagon Data Science & AI** — Nantes, juin 2026.

---

## 🎯 Objectif

Concevoir un modèle ML agile capable de **prédire le risque de sécheresse à 3 mois** sur les aquifères de Vendée et du Poitou, en anticipant le franchissement des seuils réglementaires (Vigilance / Alerte / Alerte Renforcée / Crise).

Les modèles physiques existants du BRGM (Gardénia, Tempo) offrent un suivi en temps réel mais restent lourds et peu adaptés à la prévision locale. Hydro-Sense propose une alternative légère, localisée et déployable.

---

## 🗺️ Contexte

La Vendée et le Poitou-Charentes subissent des **tensions hydriques extrêmes** sur leurs aquifères régionaux, avec des conflits d'usage récurrents (agriculture vs. écologie) et des enjeux de préservation du Marais poitevin. Le service de surveillance BRGM [Météeau Nappes](https://app.meteeaunappes.brgm.fr/desktop) fournit les données de référence.

---

## 👥 Équipe

Projet Le Wagon Nantes — Batch juin 2026

| Membre |
|--------|
| **Maxime** | Project lead |
| **Yann** |
| **Romain** |

---

## 📊 Données

### Sources principales

| Source | Type | Accès |
|--------|------|-------|
| [Hubeau — Piézométrie](https://hubeau.eaufrance.fr/page/api-niveaux-nappes) | Niveaux des nappes | REST API |
| [Météo-France](https://portail-api.meteofrance.fr) | Précipitations par département | REST API |
| [BDLISA](https://bdlisa.eaufrance.fr) | Métadonnées géologiques | Web |
| [Eaufrance — ETP](https://www.eaufrance.fr) | Évapotranspiration quotidienne | CSV |

---

## 🚀 Installation

### 1. Cloner le repo

```bash
git clone git@github.com:charourou/Projet_Hydrosense.git
cd Projet_Hydrosense
```

### 2. Créer l'environnement Python

```bash
pyenv install 3.10.6
pyenv virtualenv 3.10.6 Projet_Hydrosense
pyenv local Projet_Hydrosense
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

Crée un fichier `.env` à la racine :

```
GOOGLE_APPLICATION_CREDENTIALS= chemin absolu vers hydro-sense-498112-d8b48c4804b5.json

GOOGLE_PROJECT_ID=hydro-sense-498112
BQ_DATASET_ID=piezometry
```

---

## Lancement du Streamlit App

(1ere installation) : make install

run install (pour lancer l'app)
