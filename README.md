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
```

---

## **Requirements**

The following dependencies are required to run the project:

- **Python**: Version >= 3.7
- **TensorFlow**: Version >= 2.9
- **Flask**: Version >= 2.0
- **Keras NLP**: For text preprocessing and embedding
- Additional libraries: 
  - `Pandas`
  - `NumPy`
  - `Scikit-learn`

All dependencies are listed in the `requirements.txt` file.

---

## **Installation and Setup**

Follow these steps to set up the project:

### **Step 1**: Clone the Repository
```bash
git clone https://github.com/mohdamaj/BERT-Sentiment-Analysis.git
cd BERT-Sentiment-Analysis
```

### **Step 2**: Create and Activate a Virtual Environment
```bash
python -m venv venv
```

- **On Unix or MacOS**:
```bash
source venv/bin/activate
```
- **On Windows**:
```bash
venv\Scripts\activate
```

### **Step 3**: Install Dependencies
```bash
pip install -r requirements.txt
```

### **Step 4**: Run the Flask Application
```bash
python app.py
```
The app will run locally, and you can access it at http://127.0.0.1:5000.

---

## **Usage**

Once the Flask app is running:
1. Open the web application in your browser.
2. Enter a text snippet in the input field.
3. Click the Submit button to see the predicted emotion.

---

## **Model Training and Customization**

**Model Architecture**

The model uses the following components:

- **BERT Backbone**: Pre-trained `bert_base_en_uncased` for contextual embeddings.
- **Preprocessing**: Tokenization and input preparation using `BertPreprocessor`.
- **Classification Head**:
  - Dropout layer for regularization.
  - Dense layer with softmax activation for emotion classification.

## **Training the Model**

The trigger for training takes place in `data_ingestion.py`.

Train the model by running:

```bash
python src/pipeline/data_ingestion.py
```
The trained model will be saved in the `artifact/` directory.

---

## **Web Application**

**Features**
- **Input Form**: Allows users to enter text for classification.
- **Prediction Display**: Shows the predicted emotion on the results page.
- **Error Handling**: Provides user-friendly error messages if input is missing or invalid.

**Templates**
- **index.html**: Landing page of the web app.
- **home.html**: Displays prediction results.

---

## **Logging and Debugging**

- **Logs** are stored in the `logs/` directory.
- Centralized logging is implemented using Python’s `logging` module.
- Errors are handled via the `CustomException` class in `exception.py`.

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.








