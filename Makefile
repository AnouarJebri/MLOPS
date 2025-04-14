# Définition des variables
PYTHON = python3
VENV = venv
ACTIVATE = $(VENV)/bin/activate
REQ = requirements.txt
TRAIN_DATA = churn-bigml-80.csv
TEST_DATA = churn-bigml-20.csv
MODEL_FILE = churn_model.pkl
MAIN_SCRIPT = main.py


# Docker variables
IMAGE_NAME = anouar_jebri_4ds4
IMAGE_TAG = latest
DOCKER_USER = anouar2002
DOCKER_REPO = $(DOCKER_USER)/$(IMAGE_NAME)




# Vérifier si l'environnement virtuel est installé
check_venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "❌ L'environnement virtuel n'existe pas ! Exécutez 'make install' d'abord."; \
		exit 1; \
	fi

# Installation des dépendances
install:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r $(REQ)
	echo "✅ Environnement virtuel installé et packages installés."

# Préparation des données
prepare: check_venv
	$(ACTIVATE) && $(PYTHON) $(MAIN_SCRIPT) --data $(TRAIN_DATA)

# Entraîner le modèle
train: check_venv
	$(ACTIVATE) && $(PYTHON) $(MAIN_SCRIPT) --data $(TRAIN_DATA) --train --save $(MODEL_FILE)

# Évaluer le modèle
evaluate: check_venv
	$(ACTIVATE) && $(PYTHON) $(MAIN_SCRIPT) --data $(TEST_DATA) --evaluate --load $(MODEL_FILE)

# Vérifications CI/CD
lint: check_venv
	$(ACTIVATE) && pylint $(MAIN_SCRIPT) model_pipeline.py

format: check_venv
	$(ACTIVATE) && black $(MAIN_SCRIPT) model_pipeline.py

test: check_venv
	$(ACTIVATE) && PYTHONPATH=. pytest tests/

security: check_venv
	$(ACTIVATE) && bandit -r model_pipeline.py

# Nettoyage des fichiers temporaires
clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .mypy_cache $(MODEL_FILE)
	echo "🧹 Nettoyage effectué."

# Exécuter toutes les étapes CI
ci: lint format test security

deploy: check_venv
    $(ACTIVATE) && uvicorn app:app --reload --host 0.0.0.0 --port 8000


# Promote the last trained model to Production
promote:
	$(ACTIVATE) && $(PYTHON) -c 'from mlflow.tracking import MlflowClient; \
	client = MlflowClient(); \
	latest_version = client.get_latest_versions("RandomForestClassifier", stages=["Staging"])[0].version; \
	client.transition_model_version_stage(name="RandomForestClassifier", version=latest_version, stage="Production"); \
	print(f"🎯 Model version {latest_version} promoted to Production.")'

# Rollback the last model from Production to Staging
rollback:
	$(ACTIVATE) && $(PYTHON) -c 'from mlflow.tracking import MlflowClient; \
	client = MlflowClient(); \
	latest_version = client.get_latest_versions("RandomForestClassifier", stages=["Production"])[0].version; \
	client.transition_model_version_stage(name="RandomForestClassifier", version=latest_version, stage="Staging"); \
	print(f"🔄 Model version {latest_version} rolled back to Staging.")'

# Docker targets and some other commands to run the server
.PHONY: docker-build docker-run docker-push flask swagger

# Construire l'image Docker
docker-build:
	docker build -t $(DOCKER_REPO):$(IMAGE_TAG) .

# Exécuter le conteneur Docker
docker-run:
	docker run -p 8002:8002 $(DOCKER_REPO):$(IMAGE_TAG)

# Pousser l'image vers Docker Hub
docker-push:
	docker push $(DOCKER_REPO):$(IMAGE_TAG)

# Exécuter toutes les étapes Docker
docker: docker-build docker-push

# Exécuter l'API avec Flask
flask:
	python3 app.py

# Exécuter l'API avec FastAPI et Swagger
swagger:
	uvicorn appa:app --reload --host 0.0.0.0 --port 8001
