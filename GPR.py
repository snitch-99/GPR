import pandas as pd
import tkinter as tk
from tkinter import filedialog
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.preprocessing import StandardScaler
import numpy as np

different_file_validation = 0    # Set this to 1 if using another CSV file for validation
position_GP_model         = 0    # Set to 1 if using position as input (lat, lon), 0 otherwise
training_data             = 0.90 # Proportion of data used for training
input_column              = 4    # Column index (zero-based) for input variable
output_column             = 10   # Column index (zero-based) for output variable

def load_csv_with_dialog():
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not filepath:
        print("No file selected.")
        return None
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def GP_Reg_Scaled(input, output, validation_input):
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    input_scaled = input_scaler.fit_transform(input)
    output_scaled = output_scaler.fit_transform(output.reshape(-1, 1)).ravel()
    validation_scaled = input_scaler.transform(validation_input)

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=1e-6, normalize_y=True)
    gp.fit(input_scaled, output_scaled)

    y_pred_scaled, y_std_scaled = gp.predict(validation_scaled, return_std=True)
    y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_std = y_std_scaled * output_scaler.scale_[0]

    return np.column_stack((y_pred, y_std))

def GP_Reg_Matern_Scaled(input, output, validation_input, nu=1.5):
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    input_scaled = input_scaler.fit_transform(input)
    output_scaled = output_scaler.fit_transform(output.reshape(-1, 1)).ravel()
    validation_scaled = input_scaler.transform(validation_input)

    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e2), nu=nu)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, normalize_y=True)
    gp.fit(input_scaled, output_scaled)

    y_pred_scaled, y_std_scaled = gp.predict(validation_scaled, return_std=True)
    y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_std = y_std_scaled * output_scaler.scale_[0]

    return np.column_stack((y_pred, y_std))

def main():
    df = load_csv_with_dialog()
    if df is None:
        return

    if position_GP_model == 1:
        inputs = df.iloc[:, [1, 2]].values  # Latitude and Longitude
    else:
        inputs = df.iloc[:, [input_column]].values

    outputs = df.iloc[:, output_column].values
    rows = inputs.shape[0]

    if different_file_validation == 1:
        input_regression = inputs
        output_regression = outputs
        df_validation = pd.read_csv("06-26-25_12-25-10_students.csv")
        input_validation = df_validation.iloc[:, [1, 2]].values
        output_validation = df_validation.iloc[:, [4]].values
    else:
        split_index = math.floor(rows * training_data)
        input_regression = inputs[:split_index]
        output_regression = outputs[:split_index]
        input_validation = inputs[split_index:]
        output_validation = outputs[split_index:]

    print("Starting RBF GPR (scaled)")
    validation_output = GP_Reg_Scaled(input_regression, output_regression, input_validation)
    print("Ending RBF GPR")

    print("Starting Matern GPR (scaled)")
    validation_output_matern = GP_Reg_Matern_Scaled(input_regression, output_regression, input_validation)
    print("Ending Matern GPR")

    results = np.column_stack((
        input_validation, output_validation,
        validation_output[:, 0], validation_output[:, 1],
        validation_output_matern[:, 0], validation_output_matern[:, 1]
    ))

    result_df = pd.DataFrame(results, columns=[
        "input", "actual",
        "predicted_RBF", "variance_RBF",
        "predicted_Matern", "variance_Matern"
    ])
    result_df.to_csv("GPR_results.csv", index=False)
    print("Saved to GPR_results.csv")

if __name__ == "__main__":
    main()
