# Use an official Python image as base
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy required files to the container
COPY requirements.txt ./
COPY app.py main.py model_pipeline.py churn_model.pkl ./
COPY mlruns/0 /app/mlruns/0
COPY mlruns/models /app/mlruns/models
COPY mlruns/932258993696025042 /app/mlruns/932258993696025042


# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt

# Expose the FastAPI app on port 8002
EXPOSE 8002

# Run the FastAPI app with Flask server
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python3", "app.py"]
