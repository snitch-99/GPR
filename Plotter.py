import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import mplcursors

class SmartPlotterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart CSV Plotter (Env↔Env or Pos↔Env)")

        self.df = None

        # File picker
        ttk.Button(root, text="📂 Select CSV File", command=self.load_file).grid(row=0, column=0, columnspan=2, pady=10)

        # Input selector (multi)
        ttk.Label(root, text="Input Column(s) (X):").grid(row=1, column=0, sticky='e')
        self.input_listbox = tk.Listbox(root, selectmode='multiple', exportselection=False, height=8)
        self.input_listbox.grid(row=1, column=1)

        # Output selector (single)
        ttk.Label(root, text="Output Column (Y):").grid(row=2, column=0, sticky='e')
        self.output_combo = ttk.Combobox(root, state="readonly")
        self.output_combo.grid(row=2, column=1)

        # Plot button
        ttk.Button(root, text="📊 Plot", command=self.plot_data).grid(row=3, column=0, columnspan=2, pady=10)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return

        self.df = pd.read_csv(path)
        cols = list(self.df.columns)

        self.input_listbox.delete(0, tk.END)
        for col in cols:
            self.input_listbox.insert(tk.END, col)

        self.output_combo['values'] = cols
        self.output_combo.set('')  # clear previous selection

    def plot_data(self):
        if self.df is None:
            messagebox.showerror("Error", "No file loaded.")
            return

        selected_indices = self.input_listbox.curselection()
        input_cols = [self.input_listbox.get(i) for i in selected_indices]
        output_col = self.output_combo.get()

        if len(input_cols) == 0 or not output_col:
            messagebox.showerror("Error", "Please select at least 1 input and 1 output column.")
            return

        if len(input_cols) == 1:
            # Case 1: Env vs Env (simple scatter)
            x = self.df[input_cols[0]]
            y = self.df[output_col]

            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(x, y, c='dodgerblue', edgecolors='k', s=60, alpha=0.8)
            ax.set_xlabel(input_cols[0])
            ax.set_ylabel(output_col)
            ax.set_title(f"{output_col} vs {input_cols[0]}")
            ax.grid(True)

            cursor = mplcursors.cursor(scatter, hover=True)
            @cursor.connect("add")
            def on_add(sel):
                idx = sel.index
                sel.annotation.set_text(f"{input_cols[0]}: {x.iloc[idx]:.3f}\n{output_col}: {y.iloc[idx]:.3f}")

            plt.tight_layout()
            plt.show()

        elif len(input_cols) == 2:
            # Case 2: Position vs Env (colored scatter map)
            x1 = self.df[input_cols[0]]
            x2 = self.df[input_cols[1]]
            z = self.df[output_col]
            z_norm = (z - z.min()) / (z.max() - z.min()) if z.max() != z.min() else z * 0

            fig, ax = plt.subplots(figsize=(10, 6))
            cmap =  matplotlib.colormaps.get_cmap('coolwarm')

            norm = mcolors.Normalize(vmin=0, vmax=1)
            scatter = ax.scatter(x2, x1, c=z_norm, cmap=cmap, edgecolors='k', s=50, alpha=0.9)

            cbar = plt.colorbar(scatter)
            cbar.set_label(f"Normalized {output_col}")

            ax.set_xlabel(input_cols[1])
            ax.set_ylabel(input_cols[0])
            ax.set_title(f"{output_col} Gradient on {input_cols[0]} vs {input_cols[1]}")
            ax.grid(True)

            cursor = mplcursors.cursor(scatter, hover=True)
            @cursor.connect("add")
            def on_add(sel):
                idx = sel.index
                sel.annotation.set_text(f"{input_cols[0]}: {x1.iloc[idx]:.4f}\n{input_cols[1]}: {x2.iloc[idx]:.4f}\n{output_col}: {z.iloc[idx]:.2f}")

            plt.tight_layout()
            plt.show()

        else:
            messagebox.showerror("Error", "Please select 1 or 2 input columns only.")
            return

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartPlotterGUI(root)
    root.mainloop()
