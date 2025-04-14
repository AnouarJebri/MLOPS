from flask import Flask, request, jsonify, render_template_string
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
from model_pipeline import prepare_data
import mlflow
from sklearn.ensemble import RandomForestClassifier

# Configurer l'URI de suivi de MLflow avant d'utiliser MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger le modèle existant
MODEL_FILE = "churn_model.pkl"
try:
    model = joblib.load(MODEL_FILE)
    logger.info("✅ Modèle chargé avec succès.")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
    model = None  # Gestion si le modèle n'est pas trouvé

# Initialisation de l'application Flask
app = Flask(__name__)

# Définition du modèle de données avec Pydantic
class ClientData(BaseModel):
    Account_length: int
    Number_vmail_messages: int
    Total_day_calls: int
    Total_day_charge: float
    Total_eve_calls: int
    Total_eve_charge: float
    Total_night_calls: int
    Total_night_charge: float
    Total_intl_calls: int
    Total_intl_charge: float
    Customer_service_calls: int
    International_plan: int
    Voice_mail_plan: int
    Total_day_minutes: float
    Total_eve_minutes: float
    Total_night_minutes: float
    Total_intl_minutes: float










HTML_CONTENT = """

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Churn Prediction API Test</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #87CEEB;  /* Bleu ciel */
            color: #333;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .container {
            background-color: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            width: 90%;
            max-width: 800px;
            transition: transform 0.3s;
        }

        .container:hover {
            transform: scale(1.02);
        }

        h1 {
            text-align: center;
            color: #4CAF50; /* Couleur verte */
            margin-bottom: 20px;
            font-size: 1.8em;
        }

        .form-container {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
        }

        .form-column {
            width: 48%;
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #555;
        }

        input {
            width: 100%;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 6px;
            background: #f9f9f9;
            color: #333;
            transition: border 0.3s;
            font-size: 1em;
        }

        input:focus {
            border-color: #4CAF50; /* Couleur verte */
            outline: none;
            box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
        }

        button {
            width: 100%;
            padding: 12px;
            background-color: #4CAF50; /* Couleur verte */
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: bold;
            transition: background-color 0.3s;
        }

        button:hover {
            background-color: #45a049; /* Couleur verte plus foncée */
        }

        #result {
            margin-top: 20px;
            text-align: center;
            font-size: 1.3em;
            font-weight: bold;
            color: #4CAF50; /* Couleur verte */
        }

        @media (max-width: 600px) {
            .form-column {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Churn Prediction API Test</h1>
        <form id="predictionForm">
            <div class="form-container">
                <div class="form-column">
                    <label for="Account_length">Account Length:</label>
                    <input type="number" id="Account_length" name="Account_length" required>

                    <label for="International_plan">International Plan (0 = No, 1 = Yes):</label>
                    <input type="number" id="International_plan" name="International_plan" required>

                    <label for="Voice_mail_plan">Voicemail Plan (0 = No, 1 = Yes):</label>
                    <input type="number" id="Voice_mail_plan" name="Voice_mail_plan" required>

                    <label for="Number_vmail_messages">Number of Voicemail Messages:</label>
                    <input type="number" id="Number_vmail_messages" name="Number_vmail_messages" required>

                    <label for="Total_day_minutes">Total Day Minutes:</label>
                    <input type="number" step="0.01" id="Total_day_minutes" name="Total_day_minutes" required>

                    <label for="Total_day_calls">Total Day Calls:</label>
                    <input type="number" id="Total_day_calls" name="Total_day_calls" required>

                    <label for="Total_day_charge">Total Day Charge:</label>
                    <input type="number" step="0.01" id="Total_day_charge" name="Total_day_charge" required>
                </div>
                <div class="form-column">
                    <label for="Total_eve_minutes">Total Evening Minutes:</label>
                    <input type="number" step="0.01" id="Total_eve_minutes" name="Total_eve_minutes" required>

                    <label for="Total_eve_calls">Total Evening Calls:</label>
                    <input type="number" id="Total_eve_calls" name="Total_eve_calls" required>

                    <label for="Total_eve_charge">Total Evening Charge:</label>
                    <input type="number" step="0.01" id="Total_eve_charge" name="Total_eve_charge" required>

                    <label for="Total_night_minutes">Total Night Minutes:</label>
                    <input type="number" step="0.01" id="Total_night_minutes" name="Total_night_minutes" required>

                    <label for="Total_night_calls">Total Night Calls:</label>
                    <input type="number" id="Total_night_calls" name="Total_night_calls" required>

                    <label for="Total_night_charge">Total Night Charge:</label>
                    <input type="number" step="0.01" id="Total_night_charge" name="Total_night_charge" required>

                    <label for="Total_intl_minutes">Total International Minutes:</label>
                    <input type="number" step="0.01" id="Total_intl_minutes" name="Total_intl_minutes" required>

                    <label for="Total_intl_calls">Total International Calls:</label>
                    <input type="number" id="Total_intl_calls" name="Total_intl_calls" required>

                    <label for="Total_intl_charge">Total International Charge:</label>
                    <input type="number" step="0.01" id="Total_intl_charge" name="Total_intl_charge" required>

                    <label for="Customer_service_calls">Customer Service Calls:</label>
                    <input type="number" id="Customer_service_calls" name="Customer_service_calls" required>
                </div>
            </div>
            <button type="submit">Predict</button>
        </form>
        <div id="result"></div>
    </div>
    <script>
        document.getElementById('predictionForm').addEventListener('submit', function(event) {
            event.preventDefault();
            const formData = new FormData(event.target);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = isNaN(value) ? value : Number(value);
            });

            fetch('/predict/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').innerText = `Prediction: ${data.prediction}`;
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    """Route pour servir la page HTML."""
    return render_template_string(HTML_CONTENT)

@app.route("/predict/", methods=["POST"])
def predict():
    """Prédit si un client va faire du churn ou non."""
    if model is None:
        return jsonify({"detail": "Modèle non chargé. Réentraînez le modèle."}), 500

    try:
        data = request.json
        client_data = ClientData(**data)

        # Convertir les données en DataFrame
        df = pd.DataFrame([client_data.dict()])

        # Mapping des colonnes pour correspondre au modèle
        column_mapping = {
            "Account_length": "Account length",
            "International_plan": "International plan",
            "Voice_mail_plan": "Voice mail plan",
            "Number_vmail_messages": "Number vmail messages",
            "Total_day_calls": "Total day calls",
            "Total_day_charge": "Total day charge",
            "Total_eve_calls": "Total eve calls",
            "Total_eve_charge": "Total eve charge",
            "Total_night_calls": "Total night calls",
            "Total_night_charge": "Total night charge",
            "Total_intl_calls": "Total intl calls",
            "Total_intl_charge": "Total intl charge",
            "Customer_service_calls": "Customer service calls",
        }

        df.rename(columns=column_mapping, inplace=True)

        # Vérifier les colonnes présentes dans df
        logger.info(f"Colonnes après renommage : {df.columns.tolist()}")

        # Ajouter les colonnes manquantes avec des valeurs par défaut (si nécessaire)
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0  # Valeur par défaut, ajustez selon le besoin

        # Réorganiser les colonnes selon le modèle
        df = df[model.feature_names_in_]

        # Faire la prédiction
        prediction = model.predict(df)[0]
        message = "Churn" if prediction == 1 else "Non-Churn"

        return jsonify({"prediction": message, "message": message})

    except Exception as e:
        logger.error(f"❌ Erreur lors de la prédiction : {str(e)}")
        return jsonify({"detail": f"Erreur lors de la prédiction : {str(e)}"}), 400

class RetrainRequest(BaseModel):
    n_estimators: int
    max_depth: int
    min_samples_split: int
    train_path: str
    test_path: str

# @app.route("/retrain/", methods=["POST"])
# def retrain():
#     request_data = request.json
#     try:
#         # Préparer les données d'entraînement et de test
#         X_train, X_test, y_train, y_test = prepare_data(
#             request_data['train_path'], request_data['test_path']
#         )

#         # Créer un nouveau modèle avec les paramètres fournis
#         new_model = RandomForestClassifier(
#             n_estimators=request_data['n_estimators'],
#             max_depth=request_data['max_depth'],
#             min_samples_split=request_data['min_samples_split']
#         )
        
#         # Entraîner le modèle
#         new_model.fit(X_train, y_train)

#         # Sauvegarder le modèle réentraîné
#         joblib.dump(new_model, MODEL_FILE)

#         return jsonify({"message": "Modèle réentraîné avec succès"})

#     except FileNotFoundError as fnf:
#         return jsonify({"detail": f"Fichier de données introuvable: {fnf}"}), 404
#     except Exception as e:
#         return jsonify({"detail": f"Erreur lors du réentraînement du modèle : {e}"}), 500

if __name__ == "__main__":  # Correction ici
    app.run(host='0.0.0.0', port=8002)





    # http://localhost:8000/docs#/default/retrain_retrain__post