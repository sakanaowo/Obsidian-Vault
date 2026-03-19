---
title: "Data Preprocessing with Pandas"
aliases:
  [
    "pandas",
    "data preprocessing",
    "tiền xử lý dữ liệu",
    "missing values",
    "imputation",
  ]
tags: [concept, deep-learning, d2l, fundamentals, data, pytorch]
created: 2026-03-09
session: "D2L Tuần 2, Buổi 7 — Data Preprocessing with Pandas"
source: "D2L Chapter Preliminaries - sec_pandas"
related:
  - "[[Tensor Operations]]"
  - "[[Probability and Statistics for Deep Learning]]"
---

# Data Preprocessing with Pandas

> [!NOTE] ELI5
> Dữ liệu thực tế không bao giờ "sạch" như các tensor tổng hợp ta đã dùng. Nó đến dưới dạng file CSV, có cột bị thiếu, có kiểu string lẫn số, có outlier. **Pandas** là công cụ chuẩn để đọc, làm sạch, và biến dữ liệu "bẩn" đó thành tensor PyTorch mà model có thể ăn được. Đây là bước **hầu hết thời gian thực tế** của một ML engineer.

## 1. Tại sao Data Preprocessing quan trọng?

Trong thực tế, **data pipeline** thường chiếm 60–80% công sức của một ML project:

| Nguồn dữ liệu thực tế     | Vấn đề thường gặp                   |
| ------------------------- | ----------------------------------- |
| CSV từ database           | Missing values (`NaN`), wrong types |
| Images từ web scraping    | Corrupted files, wrong resolution   |
| Text corpus               | Encoding errors, duplicates         |
| Sensor data               | Outliers, gaps, drift               |
| Multi-table relational DB | Cần join, aggregate                 |

**Pipeline chuẩn:** Raw data → **Pandas** → Clean DataFrame → **torch.tensor()** → Model

---

## 2. Reading Data — Đọc file CSV

```python
import os
import pandas as pd

# Tạo dataset mẫu (house prices)
os.makedirs('../data', exist_ok=True)
data_file = '../data/house_tiny.csv'
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')

# Đọc
data = pd.read_csv(data_file)
print(data)
#    NumRooms RoofType   Price
# 0       NaN      NaN  127500
# 1       2.0      NaN  106000
# 2       4.0    Slate  178100
# 3       NaN      NaN  140000
```

**Pandas tự động:**

- Parse header từ dòng đầu
- Detect kiểu dữ liệu (float, string)
- Chuyển `NA` → `NaN` (Not a Number)

---

## 3. Missing Values — Giá trị bị thiếu

> [!NOTE] ELI5
> `NaN` là "hố đen" trong data — không biết giá trị thực là gì. Nếu để nguyên, model sẽ crash hoặc học sai. Có hai chiến lược: **điền vào** (imputation) hoặc **xóa đi** (deletion). Cái nào tốt hơn phụ thuộc vào domain knowledge.

### Bản chất của Missing Data — 3 loại khác nhau

| Loại     | Tên                          | Ví dụ                                              | Xử lý                       |
| -------- | ---------------------------- | -------------------------------------------------- | --------------------------- |
| **MCAR** | Missing Completely At Random | Sensor bị lỗi ngẫu nhiên                           | Deletion/Mean imputation OK |
| **MAR**  | Missing At Random            | Người thu nhập cao hay bỏ qua câu hỏi lương        | Model-based imputation      |
| **MNAR** | Missing Not At Random        | Bệnh nhân nặng bỏ follow-up → thiếu worst outcomes | Chú ý! Deletion gây bias    |

> [!WARNING] MNAR là nguy hiểm nhất
> Nếu data bị thiếu **có correlation với label** (MNAR), xóa hoặc impute naively sẽ gây **selection bias** — model học sai phân phối thực. Rất phổ biến trong medical AI và financial data.

### Chiến lược 1: Imputation (Điền vào)

**Numerical columns — Mean imputation:**

```python
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
# NumRooms: NaN → mean(2.0, 4.0) = 3.0
print(inputs)
#    NumRooms RoofType
# 0       3.0      NaN
# 1       2.0      NaN
# 2       4.0    Slate
# 3       3.0      NaN
```

**Categorical columns — One-hot với NaN category:**

```python
inputs = pd.get_dummies(inputs, dummy_na=True)
# RoofType_Slate  RoofType_nan
#      0               1          ← row 0: không phải Slate, là NaN
#      0               1
#      1               0          ← row 2: là Slate
#      0               1
```

> [!NOTE] Tại sao One-Hot Encoding?
> Model không hiểu "Slate > NaN" hay "NaN = 0". **One-hot** biến category thành binary vector — mỗi category là 1 feature riêng, không giả định thứ tự hay khoảng cách giữa các category. Đây là cách **đúng đắn về mặt toán học** để represent categorical data.

**Chiến lược khác (nâng cao):**

- **Median imputation** — robust hơn mean khi có outliers
- **Forward/backward fill** (`ffill`/`bfill`) — cho time series
- **KNN imputation** — điền theo nearest neighbor trong feature space
- **Multiple Imputation** — dùng model để predict missing values

### Chiến lược 2: Deletion (Xóa)

```python
# Xóa rows có bất kỳ NaN nào
data.dropna(axis=0)

# Xóa columns có > 50% NaN
data.dropna(axis=1, thresh=len(data)*0.5)
```

**Khi nào dùng deletion?**

- Khi số row thiếu < 5% tổng data
- Khi có đủ data để chịu được mất mát
- Khi data là MCAR (imputation sẽ không cải thiện gì)

---

## 4. Conversion to Tensor

Sau khi data sạch (toàn số), convert sang tensor PyTorch:

```python
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))

print(X)
# tensor([[3., 0., 1.],
#         [2., 0., 1.],
#         [4., 1., 0.],
#         [3., 0., 1.]], dtype=torch.float64)
print(y)
# tensor([127500., 106000., 178100., 140000.], dtype=torch.float64)
```

**Pipeline hoàn chỉnh:**

```
CSV file
  → pd.read_csv()         [DataFrame]
  → iloc (split X, y)     [DataFrame + Series]
  → get_dummies ()        [one-hot encode categoricals]
  → fillna(mean())        [impute numericals]
  → to_numpy(dtype=float) [NumPy array]
  → torch.tensor()        [PyTorch Tensor]
  → DataLoader            [batches for training]
```

---

## 5. Indexing & Selection — Truy cập data

```python
# Theo số (iloc)
data.iloc[0]          # row 0
data.iloc[:, 1]       # column 1
data.iloc[1:3, 0:2]   # rows 1-2, cols 0-1

# Theo tên (loc)
data.loc[:, 'Price']          # column "Price"
data.loc[data['Price'] > 120000]  # filter rows

# Boolean mask
mask = data['NumRooms'] > 3
data[mask]
```

---

## 6. Thực tế trong DL projects

**Vấn đề thực tế mà pandas giải quyết:**

| Tình huống                | Công cụ pandas                      |
| ------------------------- | ----------------------------------- |
| Dataset từ nhiều file CSV | `pd.concat()`, `pd.merge()`         |
| Feature engineering       | `.apply()`, `.groupby()`, `.agg()`  |
| Train/val/test split      | Boolean indexing + `reset_index()`  |
| Kiểm tra class imbalance  | `.value_counts()`                   |
| Data visualization nhanh  | `.describe()`, `.hist()`, `.corr()` |
| Export lại sau xử lý      | `to_csv()`, `to_parquet()`          |

> [!NOTE] Pandas vs Alternatives
> | | **Pandas** | **Polars** | **NumPy** |
> |---|---|---|---|
> | Tốc độ (large data) | Chậm (Python loops) | Nhanh hơn 5-10× (Rust) | Phụ thuộc op |
> | API | Mature, nhiều tài liệu | Mới, cleaner API | Low-level |
> | Missing values | `NaN` built-in | `null` built-in | Không tự nhiên |
> | Khi nào dùng | Mọi project DL cơ bản | Big data preprocessing | Tensor ops thuần |

---

## Exercises (từ D2L)

1. Load Abalone dataset từ UCI Repository. Bao nhiêu % có missing values?
2. Thử indexing bằng tên column thay vì số: `data.loc[:, 'NumRooms']`
3. Dataset lớn bao nhiêu thì bắt đầu gặp memory limit trên laptop của bạn?
4. Nếu column có **rất nhiều categories** (e.g., zip codes), one-hot sẽ tạo ra bao nhiêu features? Giải pháp thay thế?

---

> [!TODO] Mở rộng
>
> - [[Feature Engineering]] — tạo features mới từ raw data
> - [[Data Augmentation]] — tăng số lượng data bằng biến đổi
> - [[Train-Val-Test Split]] — cách chia data đúng để tránh data leakage
> - Tìm hiểu: `torch.utils.data.Dataset` + `DataLoader` — đưa pandas DataFrame vào training loop
