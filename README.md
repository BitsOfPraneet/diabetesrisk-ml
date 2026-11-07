# DiabetesRisk ML

A containerized web application that predicts diabetes risk from basic medical parameters using a Random Forest classifier. Built with Flask, scikit-learn, and Podman.

> **Disclaimer:** This project is for educational purposes only. It is not a medical diagnosis and must not be used for clinical decisions.

## Features

- Trains a Random Forest classifier on the Pima Indians Diabetes dataset
- Web form for entering 8 medical parameters
- Risk prediction with model confidence
- Displays model performance metrics (accuracy, precision, recall, F1)
- Runs in a Podman container with a custom network and persistent volume

## Tech Stack

- Python 3.11, Flask 3.0.3
- scikit-learn, pandas, NumPy, joblib
- Podman (rootless) for containerization

## Project Structure

```
diabetes_predictor/
├── app/
│   ├── app.py              # Flask web application
│   ├── train.py            # Model training script
│   ├── requirements.txt    # Python dependencies
│   └── templates/
│       ├── index.html      # Input form
│       └── result.html     # Prediction result and metrics
├── data/
│   └── diabetes.csv        # Pima Indians Diabetes dataset
├── Containerfile           # Image build instructions
├── .gitignore
└── README.md
```

## Dataset

Pima Indians Diabetes Database (768 samples, 8 features, binary outcome).
Download from Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
Place it at `data/diabetes.csv` before building.

## Prerequisites

- Linux (Ubuntu) or Windows with WSL2
- Podman

## Commands Used

### 1. Install Podman
```bash
sudo apt update && sudo apt install -y podman
podman --version
```

### 2. Create network and volume
```bash
podman network create diabetes-net
podman volume create diabetes-model-vol
```

### 3. Build the image
```bash
cd ~/diabetes_predictor
podman build -t localhost/diabetesrisk-ml:v1.2 .
podman images
```

### 4. Run the container
```bash
podman run -d \
  --name diabetes-app \
  --network diabetes-net \
  -p 5000:5000 \
  -v diabetes-model-vol:/model \
  localhost/diabetesrisk-ml:v1.0

podman ps
podman logs diabetes-app
```

### 5. Container management
```bash
podman inspect diabetes-app
podman stats diabetes-app --no-stream
podman exec -it diabetes-app bash
podman port diabetes-app
podman stop diabetes-app
podman start diabetes-app
podman rm diabetes-app
```

### 6. Push to Docker Hub
```bash
podman login docker.io
podman tag localhost/diabetesrisk-ml:v1.2 docker.io/YOURUSERNAME/diabetesrisk-ml:v1.0
podman push docker.io/YOURUSERNAME/diabetesrisk-ml:v1.0
```

### 7. Pull and run from registry
```bash
podman pull docker.io/YOURUSERNAME/diabetesrisk-ml:v1.0
podman run -d --name diabetes-app --network diabetes-net \
  -p 5000:5000 -v diabetes-model-vol:/model \
  docker.io/YOURUSERNAME/diabetesrisk-ml:v1.0
```

## Accessing the Application

Open the app in your browser:

- Native Linux: `http://localhost:5000`
- WSL2: `http://<WSL-IP>:5000` (get the IP with `hostname -I`)

## Image Versions

| Tag  | Change                                              |
|------|-----------------------------------------------------|
| v1.0 | Initial Build                                       |

## Registry

Docker Hub: https://hub.docker.com/r/praneetsorte/diabetesrisk-ml

## Troubleshooting

| Problem | Fix |
|---|---|
| `Containerfile` not found | Run `podman build` from the project root, ending with `.` |
| `short-name did not resolve` | Use the `localhost/` or `docker.io/` prefix |
| Page not reachable on `localhost` in WSL | Use the WSL IP from `hostname -I` |
| Build fails with odd errors on Windows files | Convert line endings: `dos2unix` on `.py`, `.txt`, and `Containerfile` |

## Authors

- Made by: Praneet Praful Sorte, Bornil Dutta and Nitya Kaul
- Registration No.: RA2411026010677,RA2411026010661 and RA2411026010672