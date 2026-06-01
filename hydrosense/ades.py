import os, time
from datetime import datetime
import pandas as pd

class GestionnairePiezometrie:
    def __init__(self, dossier_sortie="donnees_bassin"):
        """
        dossier de sauvegarde s'il n'existe pas.
        """
        self.dossier_sortie = dossier_sortie
        self.url_base = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques.csv"

        if not os.path.exists(self.dossier_sortie):
            os.makedirs(self.dossier_sortie)
            print(f"Dossier '{self.dossier_sortie}' créé.")


    def telecharger_forage(self, bss_id):
        """
        Télécharge l'historique complet d'un forage et le sauvegarde.

        liste des items existents :
        ['code_bss', 'bss_id', 'urn_bss', 'date_mesure', 'timestamp_mesure',
        'niveau_nappe_eau', 'mode_obtention', 'statut', 'qualification',
        'code_continuite', 'nom_continuite', 'code_producteur',
        'nom_producteur', 'code_nature_mesure', 'nom_nature_mesure',
        'profondeur_nappe']

        """
        # Verifier que la donnée est exite peut-être
        nom_fichier = f"piezo_{bss_id.replace('/', '_')}.csv"
        chemin_complet = os.path.join(self.dossier_sortie, nom_fichier)
        # print(chemin_complet)
        # print(os.path.exists(chemin_complet))


        if os.path.exists(chemin_complet):
            print(f"Info: données {bss_id} déja présente")
            return pd.read_csv(chemin_complet, sep=';')
        # early exists


        #  selection des champs OPTIONNEL
        champs_utiles = "date_mesure,niveau_nappe_eau,profondeur_nappe"
        url = f"{self.url_base}?bss_id={bss_id}&size=20000&fields={champs_utiles}"
        # Tout les champs - Attention NE PAS GARDER TOUT LES CHAMPS POUR DES sauvegardes de masse.
        if False:
            url = f"{self.url_base}?bss_id={bss_id}&size=20000&"


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        print(f"Téléchargement de {bss_id}...")
        try:
            df = pd.read_csv(url, sep=";")

            if df.empty:
                print(f"  -> Aucune donnée trouvée pour {bss_id}.")
                return False

            # Nettoyage et formatage
            if 'date_mesure' in df.columns:
                df['date_mesure'] = pd.to_datetime(df['date_mesure'])
                df = df.sort_values('date_mesure')

            # Sauvegarde en CSV local (sans l'index de pandas)
            df.to_csv(chemin_complet, index=False, sep = ';' )
            print(f"  -> Succès : {len(df)} mesures sauvegardées dans {nom_fichier}")
            return df

        except Exception as e:
            print(f"  -> Erreur lors du traitement de {bss_id} : {e}")
            return False

    def telecharger_bassin_versant(self, liste_forages, pause_secondes=3):
        """
        Traite une liste de forages.
        """

        print(f"--- Début du traitement pour {len(liste_forages)} forages ---")

        for i, bss_id in enumerate(liste_forages):
            self.telecharger_forage(bss_id)

            if i < len(liste_forages) - 1:
                print(f"  [Pause de {pause_secondes}s pour ne pas surcharger le serveur...]")
                time.sleep(pause_secondes)

        print("--- Traitement du bassin versant terminé ! ---")


if __name__ == "__main__":
    gestionnaire = GestionnairePiezometrie(dossier_sortie="donnees_saffre")

    forages_bassin = [
        "BSS001EUKK", # forage à Saffré
        "BSS001PYMM", # Mâcon (issu de la doc)
        "BSS001PYKW"  # Un autre exemple
    ]

    gestionnaire.telecharger_bassin_versant(forages_bassin,
                                            pause_secondes=3
                                            )
