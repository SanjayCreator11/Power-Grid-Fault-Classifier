An interactive web application using an Artificial Neural Network (ANN) to classify 12 different types of faults on a three-phase power transmission line. 
This tool is built with Scikit-learn and deployed with Streamlit to provide a real-time diagnostic tool for power grid analysis.

## About The Project

This project provides an end-to-end solution for automatically detecting and classifying faults in a power transmission system. By leveraging a custom dataset of electrical measurements, an Artificial Neural Network is trained to recognize the unique signatures of various fault conditions.

The primary goal is to create a practical, user-friendly tool that can assist in rapid diagnostics, thereby enhancing grid reliability. The final model is wrapped in a Streamlit web application where users can input parameters and receive an instant classification of the grid's status.

-----

## Features

  - **Accurate Fault Classification**: Utilizes a tuned Artificial Neural Network (ANN) to classify 12 distinct fault types.
  - **Feature Engineering**: Creates new, informative features (e.g., current difference) to improve model accuracy.
  - **Interactive Web Interface**: A user-friendly web app built with Streamlit allows for easy input of electrical parameters.
  - **Real-time Prediction**: Provides instant fault classification based on user inputs.
  - **Complete Workflow**: The repository includes the dataset, model training script, and the final deployment-ready application.

-----

## Project Structure

```
.
├── Fault_Data_Updated.csv         # The raw dataset used for training
├── train_model.py                 # Script to train and save the final model and scaler
├── fault_classifier_model.joblib  # The saved, pre-trained ANN model file
├── scaler.joblib                  # The saved scaler object for data normalization
├── app.py                         # The main Streamlit web application file
└── requirements.txt               # A list of all necessary Python libraries
```

-----

## How to Use

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

Make sure you have Python 3.8+ installed. You can install all the necessary libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Running Locally

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Train the Model (One-Time Step):**
    Although a pre-trained model is provided, you can retrain it by running the training script. This will generate the `.joblib` files.

    ```bash
    python train_model.py
    ```

3.  **Launch the Streamlit Web App:**
    Once the model files are present, start the Streamlit application.

    ```bash
    streamlit run app.py
    ```

4.  **Use the Application:**

      - Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
      - Use the sliders and input boxes in the sidebar to enter the electrical measurements.
      - Click the "Predict Fault Type" button to see the model's classification.

-----

## Model and Data Details

  - **Model**: `MLPClassifier` (Multi-layer Perceptron) from Scikit-learn. This is a feedforward Artificial Neural Network (ANN) with a deep architecture (`(150, 100, 50)`) designed to capture complex, non-linear patterns.
  - **Dataset**: The training data (`Fault_Data_Updated.csv`) is a self-made dataset generated from **simulations in MATLAB**. It contains nearly 5,000 instances of 12 different grid conditions.
  - **Features**: The model is trained on 19 original electrical measurements plus 3 engineered features that calculate the difference between current sent and received, which is a strong indicator of leakage faults.

-----

## Technologies Used

  - **Python**: Core programming language.
  - **MATLAB**: Used for the initial data simulation.
  - **Pandas**: For data manipulation and analysis.
  - **Scikit-learn**: For building and training the machine learning model.
  - **Streamlit**: For creating and deploying the interactive web application.
  - **Joblib**: For saving and loading the trained model.
