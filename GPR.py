import pandas as pd
import tkinter as tk
from tkinter import filedialog
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import DotProduct


different_file_validation = 0    # Set this to 1 if using another CSV file for validation
position_GP_model         = 0    # Set to 1 if using position as input (lat, lon), 0 otherwise
training_data             = 0.90 # Proportion of data used for training
input_column              = 4    # Column index (zero-based) for input variable
output_column             = 10   # Column index (zero-based) for output variable


def custom_optimizer(obj_func, initial_theta, bounds):
    theta_opt, min_val, _ = fmin_l_bfgs_b(
        obj_func, initial_theta, bounds=bounds, maxiter=300
    )
    return theta_opt, min_val

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

    print("📊 Input shape:", input_scaled.shape)
    print("📈 Output shape:", output_scaled.shape)
    print("🔍 Output mean/std:", np.mean(output_scaled), np.std(output_scaled))

    kernel = C(1.0, (1e-3, 1e4)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e2))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        alpha=1e-2,
        normalize_y=True,
        optimizer=custom_optimizer
    )

    print("🧠 Fitting Gaussian Process Regressor(RBF)...")
    gp.fit(input_scaled, output_scaled)
    print("✅ Fitting complete.")

    print("🧪 Optimized kernel:", gp.kernel_)
    print("🧮 Log Marginal Likelihood:", gp.log_marginal_likelihood(gp.kernel_.theta))

    print("📊 Predicting on validation set(RBF)...")
    y_pred_scaled, y_std_scaled = gp.predict(validation_scaled, return_std=True)
    print("✅ Prediction complete.")

    y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_std = y_std_scaled * output_scaler.scale_[0]

    return np.column_stack((y_pred, y_std))

def GP_Reg_Matern_Scaled(input, output, validation_input, nu=2.5):
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    input_scaled = input_scaler.fit_transform(input)
    output_scaled = output_scaler.fit_transform(output.reshape(-1, 1)).ravel()
    validation_scaled = input_scaler.transform(validation_input)

    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e2), nu=nu)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-2, normalize_y=True, optimizer=custom_optimizer)


    print("🧠 Fitting Gaussian Process Regressor(Maxtern)...")
    gp.fit(input_scaled, output_scaled)
    print("✅ Fitting complete.")

    print("🧪 Optimized kernel:", gp.kernel_)
    print("🧮 Log Marginal Likelihood:", gp.log_marginal_likelihood(gp.kernel_.theta))
    
    print("📊 Predicting on validation set(Matern)...")
    y_pred_scaled, y_std_scaled = gp.predict(validation_scaled, return_std=True)
    print("✅ Prediction complete.")

    y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_std = y_std_scaled * output_scaler.scale_[0]

    return np.column_stack((y_pred, y_std))

def GP_Reg_RQ_Scaled(input, output, validation_input, alpha=0.01):
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    input_scaled = input_scaler.fit_transform(input)
    output_scaled = output_scaler.fit_transform(output.reshape(-1, 1)).ravel()
    validation_scaled = input_scaler.transform(validation_input)

    # Define the Rational Quadratic kernel with low alpha (more flexible)
    kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=alpha, length_scale_bounds=(1e-3, 1e2))

    # Instantiate GPR with custom optimizer
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        alpha=1e-2,
        normalize_y=True,
        optimizer=custom_optimizer
    )

    print("🧠 Fitting Gaussian Process Regressor (Rational Quadratic)...")
    gp.fit(input_scaled, output_scaled)
    print("✅ Fitting complete.")

    print("🧪 Optimized kernel:", gp.kernel_)
    print("🧮 Log Marginal Likelihood:", gp.log_marginal_likelihood(gp.kernel_.theta))

    print("📊 Predicting on validation set (Rational Quadratic)...")
    y_pred_scaled, y_std_scaled = gp.predict(validation_scaled, return_std=True)
    print("✅ Prediction complete.")

    # Inverse transform predictions
    y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_std = y_std_scaled * output_scaler.scale_[0]

    return np.column_stack((y_pred, y_std))
def GP_Reg_Dot_Scaled(input, output, validation_input):
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    input_scaled = input_scaler.fit_transform(input)
    output_scaled = output_scaler.fit_transform(output.reshape(-1, 1)).ravel()
    validation_scaled = input_scaler.transform(validation_input)

    kernel = DotProduct(sigma_0=1.0)  # You can try other sigma_0 values

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        alpha=1e-2,
        normalize_y=True,
        optimizer=custom_optimizer
    )

    print("🧠 Fitting Gaussian Process Regressor (DotProduct)...")
    gp.fit(input_scaled, output_scaled)
    print("✅ Fitting complete.")

    print("🧪 Optimized kernel:", gp.kernel_)
    print("🧮 Log Marginal Likelihood:", gp.log_marginal_likelihood(gp.kernel_.theta))

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

    if different_file_validation == 1:
        input_regression = inputs
        output_regression = outputs
        df_validation = pd.read_csv("06-26-25_12-25-10_students.csv")
        input_validation = df_validation.iloc[:, [1, 2]].values
        output_validation = df_validation.iloc[:, [4]].values
    else:
        input_regression, input_validation, output_regression, output_validation = train_test_split(
            inputs, outputs,
            train_size=training_data,
            random_state=42,
            shuffle=True
        )

    print("Starting RBF GPR (scaled)")
    validation_output = GP_Reg_Scaled(input_regression, output_regression, input_validation)
    print("Ending RBF GPR")

    print("Starting Matern GPR (scaled)")
    validation_output_matern = GP_Reg_Matern_Scaled(input_regression, output_regression, input_validation)
    print("Ending Matern GPR")

    print("Starting RQ GPR (scaled)")
    validation_output_RQ = GP_Reg_RQ_Scaled(input_regression, output_regression, input_validation)
    print("Ending RQ GPR")

    print("Starting DotProduct GPR (scaled)")
    validation_output_dot = GP_Reg_Dot_Scaled(input_regression, output_regression, input_validation)
    print("Ending DotProduct GPR")

    results = np.column_stack((
        input_validation, output_validation,
        validation_output[:, 0], validation_output[:, 1],
        validation_output_matern[:, 0], validation_output_matern[:, 1],
        validation_output_RQ[:, 0], validation_output_RQ[:, 1],
        validation_output_dot[:, 0], validation_output_dot[:, 1]
    ))

    result_df = pd.DataFrame(results, columns=[
        "input", "actual",
        "predicted_RBF", "variance_RBF",
        "predicted_Matern", "variance_Matern",
        "predicted_RQ", "variance_RQ",
        "predicted_Dot", "variance_Dot"
    ])
    result_df.to_csv("GPR_results.csv", index=False)
    print("Saved to GPR_results.csv")

if __name__ == "__main__":
    main()
