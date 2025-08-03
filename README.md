# EnrollMate
A Stacked, Gradient-Boosting Model for Binary Classification of Clinical Enrollment Targets

Machine Learning Model for Clinical Trial Success Prediction
This repository contains the code for a machine learning model designed to predict the success of clinical trials. The model is built using the LightGBM framework and incorporates a comprehensive set of features, including demographic data, study design elements, and text-based information.

🚀 Getting Started
Follow these steps to set up the environment and run the code.

## Prerequisites
First, ensure you have Python 3.8 or newer installed. You'll need to install the required libraries. It's recommended to do this in a virtual environment.

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
Next, install the dependencies:


pip install pandas numpy scikit-learn lightgbm matplotlib seaborn shap
Note: The script also uses optuna for hyperparameter tuning. If you want to use that feature, install it as well: pip install optuna.

## Data
This model requires a dataset with specific columns for feature engineering. The script is configured to read a CSV file.

Place your dataset file in the same directory as the Python script.

Update the file path in the df = pd.read_csv("DATASET FILE PATH GOES HERE") line to point to your data file.

The script expects the following key columns to be present in the dataset:

Percentage: The target variable for trial success.

Age, Phases, Primary Outcome Measures, Start Date, Primary Completion Date, Study Design, Brief Summary, Funder Type, Conditions, Study Title, Minimum Age, Maximum Age, Sex, Healthy Volunteers, Locations, Interventions.

A range of demographic columns like 0-5_total_population, 0-5_white_population, etc., as well as median_income_median and unemployment_rate_mean for different geographical bands.

## Running the Code
The script is structured to perform a complete analysis pipeline. Simply run the Python file from your terminal:

python enrollmate.py

The script will:

Perform extensive feature engineering to create hundreds of new features.

Run a 5-fold cross-validation with a LightGBM model.

Train an ensemble model by combining the predictions from all 5 folds.

Print a final summary of the model's performance metrics (ROC AUC, PR AUC, F1 Score, etc.).

Execute an ablation study to determine the importance of each feature category and print the results.

Train a final model on the entire dataset.

Generate a SHAP summary plot to explain the model's feature importances visually.

## Output
When you run the script, you can expect a detailed output in the console showing the performance metrics for each cross-validation fold, the final ensemble results, and the ablation study's findings. The script will also generate and display the following plot:

SHAP Summary Plot: A visualization of feature importance, showing how different features contribute to the model's predictions.