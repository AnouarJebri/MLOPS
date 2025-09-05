# Customer Churn Prediction – MLOps Project

## Overview

This project demonstrates an **end-to-end MLOps workflow** for predicting **customer churn** in the telecommunications industry. It covers the full lifecycle of a machine learning project: from **data exploration and modeling** to **deployment, monitoring, and continuous integration**.

The deployed service allows businesses to predict whether a customer is likely to churn and provides monitoring tools to track model performance and system health.

---

## Objectives

* **Business**: Identify customers at risk of churn and support retention strategies.
* **Data Science**: Train and evaluate ML models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) with explainability via SHAP.
* **MLOps**: Automate, deploy, and monitor the model in production using modern tools.

---

## Project Architecture

```
Customer_Churn_Project/
  ├── data/                # Raw and processed datasets
  ├── notebooks/           # Jupyter notebooks for EDA and prototyping
  ├── src/                 # Source code (data prep, training, serving)
  ├── models/              # Saved ML models (pickle/joblib artifacts)
  ├── tests/               # Unit and integration test scripts
  ├── Dockerfile           # Containerization setup
  ├── requirements.txt     # Python dependencies
  ├── jenkinsfile          # Jenkins pipeline definition
  ├── README.md            # Project documentation
  └── Customer_churn.ipynb # Main exploratory notebook
```

---

## Tech Stack

* **Data & ML**: Python, Pandas, Scikit-learn, XGBoost, SHAP
* **Experiment Tracking**: MLflow
* **Deployment**: FastAPI, Docker (public image on Docker Hub)
* **Monitoring**:

  * **Prometheus + Grafana** → CPU/memory metrics & container monitoring
  * **Elasticsearch + Kibana** → Centralized logging & model logs visualization
* **CI/CD**: Jenkins pipeline
* **Testing**: Unit and integration tests

---

## Workflow

1. **Data Processing & Model Training**

   * Load churn dataset (`churn-bigml-80.csv` + `churn-bigml-20.csv`).
   * Train ML models and log results with **MLflow**.

2. **Deployment**

   * Serve model with **FastAPI**.
   * Containerize with **Docker**.
   * Publish public image to **Docker Hub**.

3. **CI/CD**

   * Build → Test → Deploy via **Jenkins pipeline**.
   * Run unit/integration tests before deployment.

4. **Monitoring & Logging**

   * **Prometheus + Grafana**: Resource & container metrics.
   * **Kibana + Elasticsearch**: Log aggregation & visualization.

---

## Installation

Clone the repo:

```bash
git clone https://github.com/yourusername/customer-churn-mlops.git
cd customer-churn-mlops
pip install -r requirements.txt
```

Pull the public Docker image from Docker Hub:

```bash
docker pull anouar2002/anouar_jebri_4ds4
```

Run container:

```bash
docker run -p 8000:8000 anouar2002/anouar_jebri_4ds4
```

---

## API Usage

Once running, access FastAPI docs at:

```
http://localhost:8000/docs
```

**Example request:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"account_length": 128, "international_plan": "yes", "voice_mail_plan": "no", ...}'
```

**Response:**

```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.87
}
```

---

## Monitoring

* **Grafana Dashboard** → Real-time metrics (CPU, memory, Docker).
* **Kibana Dashboard** → Log visualization from Elasticsearch.
* **MLflow UI** → Experiment tracking.

---

## Future Improvements

* Add **model drift detection**.
* Automate dataset versioning with **DVC**.
* Deploy via **Kubernetes** for scaling.
* Add **canary deployment** with Jenkins.

---

## Author

Developed by **Anouar Jebri**
GitHub: [anouar2002](https://github.com/AnouarJebri)
Contact: [anouar.jebri@gmail.com]
