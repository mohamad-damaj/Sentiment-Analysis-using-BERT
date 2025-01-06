# **BERT Sentiment Analysis**

This project implements a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model to classify text into six distinct emotion categories: **Sadness**, **Joy**, **Love**, **Anger**, **Fear**, and **Surprise**. It provides an intuitive web interface powered by Flask, enabling users to input text and obtain the predicted emotion in real-time.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Project Architecture](#project-architecture)
4. [Requirements](#requirements)
5. [Installation and Setup](#installation-and-setup)
6. [Usage](#usage)
7. [Model Training and Customization](#model-training-and-customization)
8. [Web Application](#web-application)
9. [Logging and Debugging](#logging-and-debugging)
10. [Contributing](#contributing)
11. [License](#license)

---

## **Overview**

This project combines state-of-the-art NLP techniques with a fine-tuned **BERT model** to classify emotions in text. It offers:
- **Real-time Predictions**: Through a Flask-based web app.
- **Custom Pipelines**: For ingestion, transformation, and training.
- **Pretrained BERT Backbone**: Leveraging `bert_base_en_uncased` for contextualized text embeddings.

---

## **Features**

- **Sentiment Classification**:
  - Classifies text into six emotion categories.
  - Uses a pre-trained BERT backbone for superior accuracy.
- **Interactive Web Interface**:
  - A Flask-powered interface for user interaction.
- **Pipeline Structure**:
  - Modular pipeline for easy customization and scalability.
- **Error Handling and Logging**:
  - Centralized logging and custom exception handling.

---

## **Project Architecture**

The project is structured as follows:

```plaintext
BERTMODEL/
├── artifact/                 # Stores model artifacts (e.g., weights, metadata)
├── logs/                     # Logs generated during model training and runtime
├── notebooks/                # Jupyter notebooks for exploration and experimentation
├── src/                      # Core source code
│   ├── components/           # Pipeline components: data ingestion, transformation, model trainer
│   ├── pipeline/             # Train and predict pipelines
│   ├── exception.py          # Custom exception class for debugging
│   ├── logger.py             # Logging setup and utilities
│   ├── utils.py              # Helper functions
├── templates/                # HTML templates for Flask web app
│   ├── index.html            # Landing page template
│   ├── home.html             # Prediction results page
├── app.py                    # Flask application entry point
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── setup.py                  # Setup script for project packaging
