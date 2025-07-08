import pandas as pd
import tkinter as tk
from tkinter import filedialog
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np

different_file_validation = 0    # Set this to 1 if using some other excel file for validation
position_GP_model         = 1    # Set 1 if linking position to an env parameter, 0 otherwise
training_data             = 0.90 # training and validation data split
input_column              = 1    # Column number of the input data to the GP model(actual column -1)
output_column             = 4    # Column number of the output data to the GP model(actual column -1)

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
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
def GP_Reg(input,output,validation_input):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=1e-6, normalize_y=True)
    gp.fit(input,output)
    y_pred, y_std = gp.predict(validation_input,return_std=True)
    validation_result = np.column_stack((y_pred,y_std))
    return validation_result


def main():
    df      = load_csv_with_dialog()

    if position_GP_model == 1:
        inputs  = df.iloc[:, [1, 2]].values
    else:
        inputs = df.iloc[:,[input_column]]    
   
    outputs = df.iloc[:, output_column].values
    (rows,columns) = inputs.shape
 
    if different_file_validation == 1:
        reg_index         = math.floor(rows*1)
        input_regression  = inputs[:reg_index]
        output_regression = outputs[:reg_index]
        df_validation     = pd.read_csv("06-26-25_12-25-10_students.csv")
        input_validation  = df_validation.iloc[:, [1, 2]].values
        output_validation = df_validation.iloc[:, [4]].values
    else:
        reg_index         = math.floor(rows*training_data)
        input_regression  = inputs[:reg_index]
        output_regression = outputs[:reg_index]
        input_validation  = inputs[reg_index:]
        output_validation = outputs[reg_index:]

    validation_output = GP_Reg(input_regression,output_regression,input_validation)

    results = np.column_stack((output_validation,validation_output[:,0],validation_output[:,1]))
    result_df = pd.DataFrame(results, columns=["actual","predicted","variance"])
    result_df.to_csv("GPR_results.csv",index=False)

if __name__ == "__main__":
    main()
