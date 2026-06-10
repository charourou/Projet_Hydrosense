# TODO — Éléments codés en dur à dynamiser (`piezo-dashboard-2.py`)

## 🔴 Critiques (données fictives)

- [ ] **Précipitations** (l.356-357) — `precip_7j = [2, 5, 11, 8, 3, 0, 0]` et `days_labels` fictifs → brancher sur une vraie API météo (ex. Open-Meteo) en fonction des coordonnées du piézomètre
- [ ] **Couleurs des markers sur la carte** (l.255) — valeurs des autres piézomètres mockées avec `rng.uniform(-2, 2)` → récupérer les vraies dernières mesures de chaque piézomètre depuis l'API/BQ
- [ ] **Date de mise à jour** (l.401) — `"Maj. 3 juin"` figé → remplacer par `_today.strftime("%-d %b")` (dynamique depuis la dernière mesure)
- [ ] **Date de franchissement du seuil** (l.412) — `"Seuil franchi le 28 mai"` fictif → calculer depuis l'historique la date où le statut a changé

## 🟠 Logique / comportement incorrect

- [ ] **Couleurs tendance** (l.375-379) — `_t_color` toujours rouge et `_p_color` toujours vert → les rendre dynamiques selon le signe et le contexte (nappe qui remonte = bonne nouvelle = vert)
- [ ] **Cohérence prévision** (l.303 vs l.430/454) — mini-chart limité à `min(30, ...)` jours mais les labels affichent "Prévision 90 j" → aligner le nombre de jours affichés avec le label

## 🟡 Paramétrage fixe

- [ ] **Département de la carte** (l.31) — `DEPT_CARTE = "79"` → suivre le département du piézomètre sélectionné (`dept`)
- [ ] **Centre et zoom de la carte** (l.243) — `location=[46.4, -0.3], zoom_start=8` → centrer dynamiquement sur les coordonnées du piézomètre sélectionné
- [ ] **Région dans le breadcrumb** (l.394) — `"Poitou-Charentes"` → récupérer depuis le catalogue (`nom_region` ou équivalent)
- [ ] **Type de nappe** (l.400) — `"Nappe libre"` identique pour tous → récupérer depuis le catalogue si disponible
- [ ] **Piézomètre par défaut** (l.159) — `"BSS001QHYH"` → à configurer via variable d'env ou paramètre URL

## ✅ Déjà réglé

- [x] **URL API prévision** (l.206) — `localhost:8000` → remplacé par `f"{API_URL}/predict"`
