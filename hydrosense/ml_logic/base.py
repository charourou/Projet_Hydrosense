import numpy as np
import pandas as pd

class BaselineLastYear:
    """
    Modèle Baseline Naïf (Persistance annuelle).
    Prédit pour le futur la valeur observée l'année dernière à la même période.
    """
    def __init__(self):
        self.history = None

    def fit(self, X_train, y_train):
        """Constructs the baseline using x_train and y_train"""

        self.history = pd.Series(y_train.values, index=X_train.index)
        return self

    def predict(self, X_test):
        """ retourne un y_pred de même longueur que X_test. Seulement utiliser l'index
        En allant chercher la valeur y_train à la date -1 an
        """
        if self.history is None:
            raise ValueError("Le modèle doit être entraîné (.fit) avant de pouvoir prédire.")

        y_pred = []

        for date in X_test.index:
            date_an_dernier = date - pd.DateOffset(years=1)

            # On cherche dans l'historique la date la plus proche (méthode 'nearest')

            try:
                idx_proche = self.history.index[self.history.index.get_indexer([date_an_dernier],
                                                                               method='nearest')[0]
                                                ]
                valeur_an_dernier = self.history.loc[idx_proche]
                y_pred.append(valeur_an_dernier)
            except Exception:
                y_pred.append(self.history.iloc[-1])

        return np.array(y_pred)
