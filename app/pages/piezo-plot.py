import altair as alt
import pandas as pd
import streamlit as st

from utils.bigquery import load_single_piezo_map
from hydrosense.database.bigquery import load_piezo_bq
from hydrosense.preprocess.cleaning import clean_piezo
from hydrosense.interface.main import train, preprocess, split_data, pred

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & STYLE DESIGN UI (MAQUETTE CSS CORRIGÉ)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(layout="wide")

# Le CSS cible désormais directement les blocs Streamlit pour un rendu parfait
st.markdown("""
    <style>
        /* Encapsulation globale pour le bloc indicateurs et le bloc seuils */
        .custom-card-container {
            background-color: #ffffff;
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
            margin-bottom: 24px;
            border: 1px solid #eef1f6;
        }
        
        /* Stylisation des sous-blocs de métriques Streamlit individuels */
        [data-testid="stMetric"] {
            background-color: #f8fafc !important;
            padding: 15px !important;
            border-radius: 12px !important;
            border: 1px solid #f1f5f9 !important;
        }
        
        /* Titres de section plus élégants */
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 14px;
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)

DATA_CODE_PIEZO = "BSS001QHYH"

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_trained_model_and_data(bss_id: str):
    df_raw = load_piezo_bq(bss_id)
    df_clean = clean_piezo(df_raw)
    df_ml = preprocess(df_clean)
    
    X_train, X_test, y_train, y_test = split_data(df_ml)
    model_val, _ = train(X_train, y_train, optimize=False)
    
    return model_val, df_clean, df_ml

with st.spinner("🧙‍♂️ Alignement du modèle XGBoost et chargement des données..."):
    model_val, df_clean, df_ml = get_trained_model_and_data(DATA_CODE_PIEZO)

df_piezo_unique = load_single_piezo_map(DATA_CODE_PIEZO)

# ══════════════════════════════════════════════════════════════════════════════
# TITRE DYNAMIQUE
# ══════════════════════════════════════════════════════════════════════════════
if not df_piezo_unique.empty:
    infos = df_piezo_unique.iloc[0]
    st.title(f"{infos['nom_commune']} ({infos['nom_departement']}) — {DATA_CODE_PIEZO}")
else:
    st.title(f"Courbe piézométrique & Validation — {DATA_CODE_PIEZO}")

st.write("") 

# ══════════════════════════════════════════════════════════════════════════════
# PREPARATIONS DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
forecast_val_series = pred(model_val, df_ml) 
forecast_index_val = pd.date_range(start='2026-03-01', periods=3, freq="ME")

df_pred_val = pd.DataFrame({
    "date_mesure": forecast_index_val,
    "niveau_nappe_eau": forecast_val_series.values,
    "Type": "Prédiction XGBoost (Test)"
})

df_hist = df_ml[['niveau_nappe_eau']].reset_index()
df_hist['Type'] = "Historique Réel"

deux_ans_en_arriere = pd.Timestamp("2026-05-31") - pd.Timedelta(days=365*2)
df_hist_filtre = df_hist[df_hist['date_mesure'] >= deux_ans_en_arriere]
df_total = pd.concat([df_hist_filtre, df_pred_val], ignore_index=True)

seuils_data = pd.DataFrame([
    {"y_min": 11.0, "y_max": 12.0, "Couleur": "Crise", "#color": "#f8cbad"},       
    {"y_min": 12.0, "y_max": 12.8, "Couleur": "Alerte", "#color": "#fce4d6"},      
    {"y_min": 12.8, "y_max": 13.4, "Couleur": "Surveillance", "#color": "#fff2cc"}, 
    {"y_min": 13.4, "y_max": 15.0, "Couleur": "Normal", "#color": "#e2f0d9"}        
])

# ══════════════════════════════════════════════════════════════════════════════
# ALTAIR GRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
fond_seuils = alt.Chart(seuils_data).mark_rect(opacity=0.45).encode(
    y=alt.Y('y_min:Q'),
    y2=alt.Y2('y_max:Q'),
    color=alt.Color('Couleur:N', scale=alt.Scale(
        domain=seuils_data["Couleur"].tolist(),
        range=seuils_data["#color"].tolist()
    ), title="Seuils de gestion")
)

lignes = alt.Chart(df_total).mark_line(strokeWidth=3).encode(
    x=alt.X('date_mesure:T', title='Date de mesure'),
    y=alt.Y('niveau_nappe_eau:Q', title="Niveau de la nappe (m)", scale=alt.Scale(zero=False)),
    color=alt.Color('Type:N', scale=alt.Scale(
        domain=["Historique Réel", "Prédiction XGBoost (Test)"],
        range=["#2484e5", "#2484e5"]
    ), title="Données"),
    strokeDash=alt.condition(
        alt.datum.Type == "Prédiction XGBoost (Test)",
        alt.value([6, 4]),
        alt.value([0, 0])
    )
)

chart = alt.layer(fond_seuils, lignes).properties(height=400).interactive().resolve_scale(color='independent')

# ══════════════════════════════════════════════════════════════════════════════
# DISPOSITION LAYOUT (2/3 BLOC GAUCHE — 1/3 COMPTEURS DROITE)
# ══════════════════════════════════════════════════════════════════════════════
col_gauche, col_droite = st.columns([2, 1])

with col_gauche:
    if not df_piezo_unique.empty:
        df_map = df_piezo_unique.rename(columns={'x': 'longitude', 'y': 'latitude'})
        st.map(df_map, zoom=10)
    else:
        st.warning("⚠️ Données géographiques non disponibles.")
        
    st.write("") 
    st.markdown("<p class='section-title'>📈 Historique & Prévision (Mars 2026 → Mai 2026)</p>", unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)

with col_droite:
    # ─── SECTION 1 : CARTES DES INDICATEURS (KPIS ENCAPSULÉS) ───
    st.markdown("<p class='section-title'>📊 Indicateurs</p>", unsafe_allow_html=True)
    
    derniere_valeur_reelle = df_hist_filtre['niveau_nappe_eau'].iloc[-1]
    moyenne_pred = df_pred_val['niveau_nappe_eau'].mean()
    
    # Utilisation d'un container unique avec style global appliqué en CSS
    with st.container():

        
        kpi_col1, kpi_col2 = st.columns(2)
        with kpi_col1:
            st.metric(label="NIVEAU ACTUEL", value=f"{derniere_valeur_reelle:.1f} m", delta="Normal", delta_color="normal")
        with kpi_col2:
            st.metric(label="TENDANCE 7 J", value="-0.3 m", delta="⬇ en baisse", delta_color="inverse")
            
        st.write("") # Marges internes entre les lignes de KPIs
        
        kpi_col3, kpi_col4 = st.columns(2)
        with kpi_col3:
            st.metric(label="PLUIE 7 J", value="38 mm", delta="⬆ 12 mm", delta_color="normal")
        with kpi_col4:
            st.metric(label="PRÉVISION 90 J", value=f"{moyenne_pred:.1f} m", delta="⚠ se creuse", delta_color="off")
            


    # ─── SECTION 2 : BLOC CARD DES SEUILS (HTML FIABLE) ───
    st.markdown("<p class='section-title'>⚠️ Seuils de gestion</p>", unsafe_allow_html=True)
    
    with st.container():

        
        derniere_val = df_hist_filtre['niveau_nappe_eau'].iloc[-1]
        statut_actuel = "Normal"
        couleur_statut = "#2ca02c"
        
        if readiness_val := derniere_val:
            if readiness_val <= 12.0:
                statut_actuel, couleur_statut = "Crise", "#d9534f"
            elif readiness_val <= 12.8:
                statut_actuel, couleur_statut = "Alerte", "#f0ad4e"
            elif readiness_val <= 13.4:
                statut_actuel, couleur_statut = "Surveillance", "#f0de4e"

        st.markdown(
            f"<div style='display: flex; justify-content: space-between; font-weight: bold; margin-bottom: 12px; font-size: 0.9rem; color: #666;'>"
            f"<span>Niveau mesuré</span>"
            f"<span style='color: {couleur_statut}; background-color: {couleur_statut}15; padding: 2px 10px; border-radius: 20px;'>{derniere_val:.1f} m · {statut_actuel}</span>"
            f"</div>", 
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style='height: 6px; width: 100%; border-radius: 3px; 
                 background: linear-gradient(to right, #f8cbad 0%, #f8cbad 25%, #fce4d6 25%, #fce4d6 50%, #fff2cc 50%, #fff2cc 75%, #e2f0d9 75%, #e2f0d9 100%); 
                 margin-bottom: 18px;'>
            </div>
            """, 
            unsafe_allow_html=True
        )

        for row in seuils_data.iloc[::-1].itertuples():
            is_actif = (statut_actuel == row.Couleur)
            tag_actif = "<span style='color: #2ca02c; font-size: 0.8em; font-weight: bold; background-color: #2ca02c15; padding: 1px 6px; border-radius: 10px;'>Actif</span>" if is_actif else "<span style='color: #ccc;'>—</span>"
            
            if row.Couleur == "Normal":
                limite_txt = f"≥ {row.y_min:.1f} m"
            elif row.Couleur == "Crise":
                limite_txt = f"≤ {row.y_max:.1f} m"
            else:
                limite_txt = f"{row.y_min:.1f} m"

            st.markdown(
                f"<div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 11px; font-size: 0.9rem;'>"
                f"  <div style='display: flex; align-items: center;'>"
                f"    <span style='height: 10px; width: 10px; background-color: {row._4}; border-radius: 50%; display: inline-block; margin-right: 10px;'></span>"
                f"    <span style='font-weight: {'600' if is_actif else 'normal'}; color: {'#000' if is_actif else '#555'};'>{row.Couleur}</span>"
                f"  </div>"
                f"  <span style='color: #777; font-family: monospace;'>{limite_txt}</span>"
                f"  <span style='width: 45px; text-align: right;'>{tag_actif}</span>"
                f"</div>", 
                unsafe_allow_html=True
            )
  

# ══════════════════════════════════════════════════════════════════════════════
# INTERFACE — METRIQUES FINALES (BAS DE PAGE)
# ══════════════════════════════════════════════════════════════════════════════
st.write("---")
st.markdown("<p class='section-title'>📊 Détail des valeurs prédites pour le jeu de test</p>", unsafe_allow_html=True)
cols_val = st.columns(3)

for i, row in enumerate(df_pred_val.itertuples()):
    with cols_val[i]:
        st.metric(label=row.date_mesure.strftime("%B %Y"), value=f"{row.niveau_nappe_eau:.2f} m")