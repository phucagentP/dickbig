import argparse
import csv
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

print("START")

def load_series_from_file(file_path, target_column=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
        if target_column:
            if target_column not in df.columns:
                raise ValueError(f"Target column not found: {target_column}")
            series = df[target_column].dropna().astype("float32")
        else:
            series = df.iloc[:, 0].dropna().astype("float32")
        return series.to_numpy()

    with open(file_path, "r", newline="") as f:
        sample = f.read(1024)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample)

        if target_column:
            reader = csv.DictReader(f)
            data = []
            for row in reader:
                if target_column in row and row[target_column] != "":
                    data.append(float(row[target_column]))
            return np.array(data, dtype=np.float32)

        if has_header:
            next(f, None)

        reader = csv.reader(f)
        data = []
        for row in reader:
            if not row:
                continue
            data.append(float(row[0]))
        return np.array(data, dtype=np.float32)


parser = argparse.ArgumentParser(description="Train LSTM with a dataset")
parser.add_argument("--csv", dest="csv_path", default=None, help="Path to CSV/XLSX dataset")
parser.add_argument("--target", dest="target_column", default=None, help="Target column name (optional)")
parser.add_argument("--window", dest="window", type=int, default=10, help="Sequence window size")
parser.add_argument("--epochs", dest="epochs", type=int, default=5, help="Training epochs")
parser.add_argument("--value", dest="user_value", type=float, default=None, help="Actual value to evaluate loss")
args = parser.parse_args()

if args.csv_path:
    data = load_series_from_file(args.csv_path, args.target_column)
else:
    # Fallback data (giong du lieu gio) neu chua truyen dataset
    data = np.sin(np.linspace(0, 50, 200)).astype(np.float32)

# Tạo chuỗi thời gian
X = []
y = []

for i in range(args.window, len(data)):
    X.append(data[i - args.window:i])
    y.append(data[i])

X = np.array(X)
y = np.array(y)

# reshape cho LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Model LSTM
model = Sequential([
    LSTM(50, input_shape=(args.window, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X, y, epochs=args.epochs, verbose=1)

# Evaluate single value against prediction (prompt if missing)
last_window = X[-1].reshape(1, args.window, 1)
pred = float(model.predict(last_window, verbose=0)[0][0])

if args.user_value is None:
    try:
        raw = input("Nhap gia tri thuc te (y) de tinh loss: ").strip()
        args.user_value = float(raw)
    except Exception:
        print("Khong nhap gia tri hop le, bo qua tinh loss.")
        args.user_value = None

if args.user_value is not None:
    mse = float(np.mean((pred - args.user_value) ** 2))
    print(f"PRED={pred:.4f} ACTUAL={args.user_value:.4f} MSE={mse:.4f}")

print("DONE")


