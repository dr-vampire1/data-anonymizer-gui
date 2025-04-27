import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from faker import Faker
import os

fake = Faker()

def apply_differential_privacy(data, epsilon):
    sensitivity = np.max(data) - np.min(data)
    noise = np.random.laplace(0, sensitivity / epsilon, size=len(data))
    return data + noise

def generalize_value(val, level):
    val_str = str(val)
    return val_str[:level] + '*' * (len(val_str) - level)

def synthetic_value(col_name):
    if "name" in col_name.lower():
        return fake.name()
    elif "city" in col_name.lower():
        return fake.city()
    elif "email" in col_name.lower():
        return fake.email()
    else:
        return fake.word()

def generate_synthetic_dataset(columns, rows=50):
    data = {}
    for col in columns:
        if "name" in col.lower():
            data[col] = [fake.name() for _ in range(rows)]
        elif "city" in col.lower():
            data[col] = [fake.city() for _ in range(rows)]
        elif "email" in col.lower():
            data[col] = [fake.email() for _ in range(rows)]
        else:
            data[col] = [np.random.randint(10, 100) for _ in range(rows)]
    return pd.DataFrame(data)

class AnonymizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Data Anonymizer GUI")
        master.configure(bg='#f0f0f0')

        self.file_path = None
        self.df = None

        # Frames
        self.top_frame = tk.Frame(master, bg='#f0f0f0')
        self.top_frame.pack(pady=10)

        self.middle_frame = tk.Frame(master, bg='#f0f0f0')
        self.middle_frame.pack(pady=10)

        self.bottom_frame = tk.Frame(master, bg='#f0f0f0')
        self.bottom_frame.pack(pady=10)

        # Top Frame Widgets
        self.upload_button = tk.Button(self.top_frame, text="Upload Excel File", command=self.upload_file)
        self.upload_button.grid(row=0, column=0, padx=5)

        self.synthetic_button = tk.Button(self.top_frame, text="Generate Synthetic Dataset", command=self.generate_synthetic)
        self.synthetic_button.grid(row=0, column=1, padx=5)

        self.columns_label = tk.Label(self.top_frame, text="Columns will appear here after upload", bg='#f0f0f0')
        self.columns_label.grid(row=1, column=0, columnspan=2, pady=5)

        # Middle Frame Widgets
        tk.Label(self.middle_frame, text="Numerical Columns (comma separated):", bg='#f0f0f0').grid(row=0, column=0, sticky='e')
        self.numerical_entry = tk.Entry(self.middle_frame, width=40)
        self.numerical_entry.grid(row=0, column=1, padx=5)

        tk.Label(self.middle_frame, text="String Columns (comma separated):", bg='#f0f0f0').grid(row=1, column=0, sticky='e')
        self.string_entry = tk.Entry(self.middle_frame, width=40)
        self.string_entry.grid(row=1, column=1, padx=5)

        tk.Label(self.middle_frame, text="k-Anonymity Value (k):", bg='#f0f0f0').grid(row=2, column=0, sticky='e')
        self.k_entry = tk.Entry(self.middle_frame, width=10)
        self.k_entry.grid(row=2, column=1, sticky='w')

        # Bottom Frame Widgets
        self.anonymize_button = tk.Button(self.bottom_frame, text="Anonymize Data", command=self.anonymize_data)
        self.anonymize_button.pack(pady=10)

    def upload_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if self.file_path:
            self.df = pd.read_excel(self.file_path)
            self.columns_label.config(text=f"Columns: {list(self.df.columns)}")
            messagebox.showinfo("File Loaded", f"Loaded {os.path.basename(self.file_path)} successfully!")

    def anonymize_data(self):
        if self.df is None:
            messagebox.showerror("Error", "No file uploaded!")
            return

        numerical_cols = [col.strip() for col in self.numerical_entry.get().split(",") if col.strip()]
        string_cols = [col.strip() for col in self.string_entry.get().split(",") if col.strip()]
        try:
            k = int(self.k_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid value for k!")
            return

        epsilon_dict = {}
        for col in numerical_cols:
            if col in self.df.columns:
                epsilon_window = tk.Toplevel(self.master)
                epsilon_window.title(f"Set Epsilon for {col}")
                tk.Label(epsilon_window, text=f"Enter ε (epsilon) for {col}: ").pack(padx=10, pady=10)
                epsilon_entry = tk.Entry(epsilon_window)
                epsilon_entry.pack(padx=10, pady=10)
                def save_epsilon():
                    epsilon = float(epsilon_entry.get())
                    epsilon_dict[col] = epsilon
                    epsilon_window.destroy()
                tk.Button(epsilon_window, text="OK", command=save_epsilon).pack(pady=10)
                epsilon_window.grab_set()
                self.master.wait_window(epsilon_window)

        # Apply differential privacy
        for col in numerical_cols:
            if col in self.df.columns:
                self.df[col] = apply_differential_privacy(self.df[col], epsilon_dict[col])

        # Find risky rows
        group_sizes = self.df.groupby(string_cols).size().reset_index(name='count')
        self.df = self.df.merge(group_sizes, on=string_cols, how='left')
        risky_rows = self.df['count'] < k

        # Anonymization method selection per column
        for col in string_cols:
            if col in self.df.columns:
                method_window = tk.Toplevel(self.master)
                method_window.title(f"Select Anonymization for {col}")
                tk.Label(method_window, text=f"Anonymization for {col}:").pack(padx=10, pady=10)
                method_var = tk.StringVar(method_window)
                method_var.set("Suppression")
                method_menu = ttk.Combobox(method_window, textvariable=method_var, values=["Suppression", "Generalization", "Synthetic Replacement"], state='readonly')
                method_menu.pack(padx=10, pady=10)

                def save_method():
                    choice = method_var.get()
                    if choice == "Suppression":
                        self.df[col] = [val if not risk else '*' for val, risk in zip(self.df[col], risky_rows)]
                    elif choice == "Generalization":
                        level_window = tk.Toplevel(self.master)
                        tk.Label(level_window, text="Enter generalization level:").pack(padx=10, pady=10)
                        level_entry = tk.Entry(level_window)
                        level_entry.pack(padx=10, pady=10)
                        def save_level():
                            level = int(level_entry.get())
                            self.df[col] = [val if not risk else generalize_value(val, level) for val, risk in zip(self.df[col], risky_rows)]
                            level_window.destroy()
                        tk.Button(level_window, text="OK", command=save_level).pack(pady=10)
                        level_window.grab_set()
                        self.master.wait_window(level_window)
                    elif choice == "Synthetic Replacement":
                        fake_vals = [synthetic_value(col) for _ in range(len(self.df))]
                        self.df[col] = [val if not risk else fake_val for val, fake_val, risk in zip(self.df[col], fake_vals, risky_rows)]
                    method_window.destroy()

                tk.Button(method_window, text="OK", command=save_method).pack(pady=10)
                method_window.grab_set()
                self.master.wait_window(method_window)

        self.df.drop(columns='count', inplace=True)
        output_file = "anonymized_output.xlsx"
        self.df.to_excel(output_file, index=False)
        messagebox.showinfo("Success", f"Anonymized data saved as {output_file}!")

    def generate_synthetic(self):
        if self.df is None:
            messagebox.showerror("Error", "No file uploaded!")
            return

        synthetic_df = generate_synthetic_dataset(list(self.df.columns), rows=50)
        synthetic_file = "synthetic_dataset.xlsx"
        synthetic_df.to_excel(synthetic_file, index=False)
        messagebox.showinfo("Success", f"Synthetic dataset saved as {synthetic_file}!")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnonymizerApp(root)
    root.geometry("700x600")
    root.mainloop()
