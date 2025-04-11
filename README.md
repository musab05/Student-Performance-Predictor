# Student Performance Predictor

This repository contains a hybrid neuro-fuzzy model that predicts student performance based on various factors, including study habits, engagement metrics, and demographics.

## Overview

The **Student Performance Predictor** uses a combination of neural networks and fuzzy logic to create a comprehensive model that accurately predicts student performance scores based on multiple input factors.

## Features

- Preprocesses data from multiple educational datasets.
- Implements a hybrid neuro-fuzzy system for predictions.
- Provides an interactive Streamlit web interface for easy predictions.
- Handles missing data and provides graceful fallbacks.

## Installation

Follow these steps to set up the project on your local machine:

1. **Clone this repository:**
   ```bash
   git clone https://github.com/musab05/Student-Performance-Predictor.git
   cd Student-Performance-Predictor
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the model (if needed):**
   ```bash
   python model.py
   ```

2. **Run the web application:**
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`).

## Data

The model uses three educational datasets:

- `xAPI-Edu-Data.csv`: Contains educational behavioral data.
- `StudentPerformanceFactors.csv`: Contains various factors affecting student performance.
- `student-scores.csv`: Contains student scores across different subjects.

Place these files in a `data/` directory relative to the project root.

## Model Details

The system uses:

- A **fuzzy logic system** for rule-based performance assessment.
- A **neural network** for learning complex patterns in student data.
- A **hybrid approach** that combines the strengths of both methods.

## Requirements

Install the dependencies specified in `requirements.txt` to ensure compatibility.

## Code Structure

- `model.py`: Handles data preprocessing, model creation, and training.
- `app.py`: Contains the Streamlit web application for predictions.
- `data/`: Directory containing the datasets.
- `scaler.pkl`: Saved scaler model for feature normalization.
- `hybrid_model.h5`: Saved hybrid neural network model.
- `fuzzy_model.pkl`: Saved fuzzy logic system.

---

Start predicting student performance today with this powerful and intuitive tool!
