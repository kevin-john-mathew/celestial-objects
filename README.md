# Celestial Bodies Detection

A Flask web application that uses a Convolutional Neural Network (CNN) to classify images of celestial bodies — planets, moons, asteroids, and galaxies — and returns facts about the predicted object pulled from Wikipedia.

## Features

- Upload an image or provide a URL to classify a celestial body
- CNN-based image classification (TensorFlow / Keras)
- Prediction confidence scores for every class
- Automatic lookup of relevant facts and statistics from Wikipedia
- Reverse image search shortcut for learning more about the result

## Supported Classes

Asteroids, Earth, Elliptical Galaxy, Jupiter, Mars, Mercury, Moon, Neptune, Saturn, Spiral Galaxy, Uranus, Venus

## Tech Stack

- **Backend:** Flask, Flask-WTF, Flask-Uploads
- **Machine Learning:** TensorFlow, Keras
- **Data:** Wikipedia API, PyYAML

## Getting Started

### 1. Create a virtual environment (recommended)

Python virtual environments keep this project's dependencies isolated from your system packages.

```bash
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate   # sh, bash, ksh, or zsh
```

If you prefer conda:

```bash
conda create -n tensorflow python=3.7
conda activate tensorflow
```

### 2. Install dependencies

Move into the image training directory and install from the requirements file:

```bash
cd hub/examples/image_retraining
pip install -r requirements.txt
```

Or install the core packages individually:

```bash
pip install tensorflow==1.14
pip install wikipedia
pip install PyYAML
```

### 3. Run the app

From the project root:

```bash
python run.py
```

The app will be available locally at `http://127.0.0.1:5000`.

## Project Structure

The web app is built with Flask. Static assets (CSS, JS, images), templates, and the `views.py` route handlers live in the [app/](/Users/kevinmathew/Documents/celestial-objects/CNN-Image-Detection-For-Celestial-Bodies/app) directory.

### Endpoints

| Route | Description |
|---|---|
| `/` | Home page — upload an image or enter a URL |
| `/result` | Displays the classification result, confidence scores, and object facts |
| `/redirectToGoogle` | Redirects to a Google reverse image search |
| `/about` | About the project |

## License

This project is open source and available for educational use.
