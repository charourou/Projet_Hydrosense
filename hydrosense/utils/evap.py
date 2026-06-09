import math
import numpy as np

SIGMA = 4.903e-9 # Constante de Stefan-Boltzmann


def estimer_precip_utile(P_mm, T_moy_C, vent_m_s, lat_deg, altitude_m, jour_annee, humidite_relative=60.0):
    """
    Estime la Précipitation Utile (PU) quotidienne en utilisant une version
    simplifiée de l'équation de Penman-Monteith (FAO-56).

    Paramètres :
    - P_mm : Précipitation du jour (mm)
    - T_moy_C : Température moyenne du jour (°C)
    - vent_m_s : Vitesse du vent à 2m de hauteur (m/s)
    - lat_deg : Latitude du piézomètre (degrés décimaux)
    - altitude_m : Altitude du piézomètre (mètres)
    - jour_annee : Jour de l'année (1 à 365/366)
    - humidite_relative : Humidité relative moyenne (%) - estimée à 60% par défaut en France.

    Retourne : un dictionnaire avec ET0 (mm) et PU (mm)
    """

    # 1. Constantes et pression atmosphérique
    P_atm = 101.3 * ((293 - 0.0065 * altitude_m) / 293) ** 5.26
    gamma = 0.000665 * P_atm  # Constante psychrométrique (kPa/°C)

    # 2. Pressions de vapeur (kPa) et pente de la courbe (delta)
    e_s = 0.6108 * math.exp(17.27 * T_moy_C / (T_moy_C + 237.3)) # Pression saturante
    e_a = e_s * (humidite_relative / 100.0)                      # Pression réelle
    delta = (4098 * e_s) / ((T_moy_C + 237.3) ** 2)              # Pente

    # 3. Rayonnement astronomique (Ra) basé sur la latitude et la date
    lat_rad = math.radians(lat_deg)
    dr = 1 + 0.033 * math.cos(2 * math.pi * jour_annee / 365)
    declinaison_sol = 0.409 * math.sin(2 * math.pi * jour_annee / 365 - 1.39)
    omega_s = math.acos(max(-1.0, min(1.0, -math.tan(lat_rad) * math.tan(declinaison_sol))))

    Ra = (24 * 60 / math.pi) * 0.0820 * dr * (
        omega_s * math.sin(lat_rad) * math.sin(declinaison_sol) +
        math.cos(lat_rad) * math.cos(declinaison_sol) * math.sin(omega_s)
    )

    # 4. Rayonnement net (Rn) - Approximations FAO pour ciel moyen
    # Rs (Rayonnement global) estimé grossièrement à 50% du rayonnement max (Ra)
    Rs = 0.50 * Ra
    Rns = (1 - 0.23) * Rs # Albedo de 0.23 pour une culture de référence (herbe)


    Rnl = SIGMA * ((T_moy_C + 273.16)**4) * (0.34 - 0.14 * math.sqrt(e_a)) * (1.35 * (Rs / (0.75 * Ra)) - 0.35)
    Rn = Rns - Rnl

    # Équation finale de Penman-Monteith
    terme_radiatif = 0.408 * delta * Rn
    terme_advectif = gamma * (900 / (T_moy_C + 273.16)) * vent_m_s * (e_s - e_a)
    denominateur = delta + gamma * (1 + 0.34 * vent_m_s)

    ET0 = max(0, (terme_radiatif + terme_advectif) / denominateur)

    # Bilan hydrique : Précipitation utile

    PU = P_mm - ET0

    return round(PU, 3)
