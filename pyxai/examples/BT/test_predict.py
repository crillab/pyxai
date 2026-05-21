import xgboost as xgb
import numpy as np

from pyxai import Learning, Explaining, Tools
from sklearn.datasets import make_regression

# 1. Générer un dataset synthétique pour la régression
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# 2. Calculer le base score (moyenne de y)
base_score = np.mean(y)
print("Base score (moyenne de y) :", base_score)

# 3. Initialiser et entraîner un modèle XGBoost pour la régression
model = xgb.XGBRegressor(
    n_estimators=3,  # 3 arbres pour l'exemple
    learning_rate=0.1,
    max_depth=2,
    objective="reg:squarederror",  # Erreur quadratique moyenne
    base_score=base_score,  # On fixe explicitement le base score
    random_state=42
)
model.fit(X, y)

# 4. Prédire pour un échantillon
sample = X[0:1]  # Premier échantillon
prediction = model.predict(sample)
print("Prédiction finale :", prediction)

# 5. Afficher les contributions de chaque arbre
# Récupérer les prédictions intermédiaires (base score + arbre par arbre)
def predict_step_by_step(model, X):
    booster = model.get_booster()
    predictions = np.repeat(model.base_score, X.shape[0])
    tree_pred = booster.predict(xgb.DMatrix(X), pred_leaf=True)
    print("tree_pred:", tree_pred)
    #for i in range(model.n_estimators):
    #    tree_pred = booster.predict(xgb.DMatrix(X), pred_leaf=True)[:, i]
    #    print("tree_pred:", tree_pred)
    #    # Pour simplifier, on suppose que chaque feuille ajoute une valeur moyenne (en pratique, c'est plus complexe)
    #    # Ici, on affiche juste l'évolution de la prédiction
    #    predictions += model.learning_rate * tree_pred
    #    print(f"Après l'arbre {i+1}, prédiction = {predictions[0]:.4f}")

print("ess:", model.get_booster().get_dump(dump_format="json"))


predict_step_by_step(model, sample)

learner, model = Learning.ModelIO.import_models(model, instances_type="tabular")
print(sample)
prediction_pyxai = model.predict_instance(sample[0])
print("prediction_pyxai: ", prediction_pyxai)