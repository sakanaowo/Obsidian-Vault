---
tags:
  - AI/Training
  - AI/Math
  - Concept
alias:
  - Objective Function
  - Cost Function
created: 2026-01-04
---

### Định nghĩa

**Loss Function** (Hàm mất mát), hay còn gọi là Cost Function hoặc Objective Function, là một hàm toán học dùng để đo lường "độ sai lệch" giữa dự đoán của mô hình ($\hat{y}$) và giá trị thực tế ($y$).

Mục tiêu của quá trình huấn luyện (Training) là tìm bộ tham số $\theta$ sao cho giá trị Loss này là nhỏ nhất (Minimization).

### Phân loại Loss Functions

Dựa trên loại bài toán, chúng ta có các nhóm hàm mất mát chính:

#### 1. Divergence-based Loss (Dùng cho Phân phối Xác suất)
Thường dùng trong bài toán Phân loại (Classification) hoặc Mô hình ngôn ngữ (Language Modeling), nơi output là xác suất.

*   **Cross-Entropy Loss (Log Loss):**
    *   *Công thức:* $L = -\sum y_{gold} \cdot \log(y_{pred})$
    *   *Ý nghĩa:* Đo độ khác biệt giữa hai phân phối xác suất. Nếu dự đoán đúng (xác suất cao cho đúng nhãn), Loss thấp. Nếu dự đoán sai tự tin (xác suất cao cho sai nhãn), Loss rất cao (phạt nặng).
    *   *Ứng dụng:* Phân loại văn bản, Dịch máy, Next-word prediction.

*   **KL Divergence Loss:**
    *   Dùng khi muốn mô hình học ra một phân phối $Q$ giống với phân phối chuẩn $P$ (ví dụ trong VAE).

#### 2. Distance-based Loss (Dùng cho Regression)
Thường dùng khi output là giá trị thực (Real value).

*   **Mean Squared Error (MSE - L2 Loss):**
    *   *Công thức:* $L = \frac{1}{n} \sum (y_{pred} - y_{gold})^2$
    *   *Đặc điểm:* Nhạy cảm với nhiễu (outliers) vì bình phương lỗi sẽ khuếch đại sai số lớn.
*   **Mean Absolute Error (MAE - L1 Loss):**
    *   *Công thức:* $L = \frac{1}{n} \sum |y_{pred} - y_{gold}|$
    *   *Đặc điểm:* Ít nhạy cảm với nhiễu hơn MSE.

#### 3. Margin-based Loss (Dùng cho Classification/Ranking)
*   **Hinge Loss:**
    *   *Công thức:* $L = \max(0, 1 - y_{gold} \cdot y_{pred})$ (với $y \in \{-1, 1\}$).
    *   *Ý nghĩa:* Chỉ phạt khi mô hình dự đoán sai hoặc dự đoán đúng nhưng chưa đủ "tự tin" (chưa vượt qua margin).
    *   *Ứng dụng:* Support Vector Machines (SVM).

#### 4. Ranking-based Loss
*   **Pairwise Ranking Loss:** So sánh từng cặp (pair) để đảm bảo mục đúng (positive) có điểm cao hơn mục sai (negative).
*   **Contrastive Loss:** Kéo các vector giống nhau lại gần, đẩy các vector khác nhau ra xa (Dùng trong [[CLIP]], Embedding training).

### Vai trò trong NLP

Trong NLP hiện đại, **Cross-Entropy** là "vua" vì hầu hết các bài toán (từ phân loại đến sinh văn bản) đều được mô hình hóa dưới dạng dự đoán phân phối xác suất của từ tiếp theo hoặc nhãn của văn bản.
