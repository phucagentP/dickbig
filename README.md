# LSTM - Flow

Flow la luong chay cua chuong trinh (cac buoc theo thu tu).

## Flow (diengio.py)

1) Doc dataset (CSV/XLSX) neu co `--csv`, neu khong thi dung du lieu gia lap.
2) Tao chuoi thoi gian `X, y` theo `--window`.
3) Build model LSTM.
4) Train voi `--epochs`.
5) Du doan tu cua so cuoi.
6) Neu khong truyen `--value` thi hoi nhap gia tri thuc te.
7) Tinh loss (MSE) cho gia tri vua nhap.
8) In `DONE`.

## Lenh chay mau

- Chay voi dataset Excel (cot cong suat):
  ```bash
  e:/Phuc/Tu-o-C/CODE-ALL-THINGS/hack/.venv/Scripts/python.exe e:/Phuc/Tu-o-C/CODE-ALL-THINGS/LSTM/diengio.py --csv e:/Phuc/Tu-o-C/CODE-ALL-THINGS/LSTM/set2.xlsx --target "Potencia de energía eólica(kW)" --epochs 5
  ```

- Neu muon truyen gia tri thuc te de tinh loss:
  ```bash
  e:/Phuc/Tu-o-C/CODE-ALL-THINGS/hack/.venv/Scripts/python.exe e:/Phuc/Tu-o-C/CODE-ALL-THINGS/LSTM/diengio.py --csv e:/Phuc/Tu-o-C/CODE-ALL-THINGS/LSTM/set2.xlsx --target "Potencia de energía eólica(kW)" --epochs 5 --value 120
  ```
