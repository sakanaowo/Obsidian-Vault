# Diagnostic Test — AI Tutor Calibration

Bộ câu hỏi chẩn đoán trình độ để tự động điều chỉnh cấu hình học tập.

## Instructions for the Tutor

1. Present questions **one category at a time**, starting from Category 1.
2. After each category, evaluate whether the student answered ≥ 60% correctly.
3. If the student scores < 60% in a category, **stop** — their depth level is the previous category.
4. If the student completes all categories, their depth is Ph.D level.
5. After calibration, update the student profile and explain the result.

## Scoring → Depth Mapping

| Highest Category Passed | Recommended Depth |
| ----------------------- | ----------------- |
| None (failed Cat. 1)    | Elementary        |
| Category 1 only         | High School       |
| Category 1 + 2          | Undergraduate     |
| Category 1 + 2 + 3      | Graduate          |
| Category 1 + 2 + 3 + 4  | Master's          |
| All categories          | Ph.D              |

## Category 1 — Nền tảng Toán (Math Foundations)

**Yêu cầu:** Trả lời bằng lời giải thích ngắn, không chỉ đáp án.

1. **Đạo hàm cơ bản:** Tính đạo hàm của $f(x) = 3x^2 + 2x - 5$. Giải thích ý nghĩa hình học của đạo hàm tại $x = 1$.

2. **Ma trận:** Cho $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ và $B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$, tính $A \times B$. Tại sao phép nhân ma trận không giao hoán?

3. **Xác suất:** Nếu P(A) = 0.3, P(B) = 0.4, P(A ∩ B) = 0.1, tính P(A|B). Giải thích bằng lời tại sao P(A|B) ≠ P(A).

4. **Vector:** Giải thích dot product giữa hai vector có ý nghĩa gì. Khi nào dot product = 0?

## Category 2 — ML Cơ bản (ML Basics)

1. **Bias-Variance Tradeoff:** Giải thích sự khác biệt giữa bias cao và variance cao. Cho ví dụ một model bị mỗi loại.

2. **Gradient Descent:** Mô tả thuật toán gradient descent bằng lời. Tại sao learning rate quá lớn gây vấn đề? Tại sao quá nhỏ cũng gây vấn đề?

3. **Overfitting:** Bạn có một model đạt 99% accuracy trên training set nhưng chỉ 60% trên test set. Chẩn đoán vấn đề và đề xuất 3 giải pháp cụ thể.

4. **Loss Function:** Tại sao dùng cross-entropy loss cho classification thay vì MSE? Giải thích từ góc độ gradient.

## Category 3 — DL Core

1. **Backpropagation:** Giải thích chain rule trong backpropagation. Cho một network đơn giản 2 lớp (input → hidden → output), viết công thức cập nhật weight cho hidden layer.

2. **CNN:** Giải thích tại sao Convolutional layer hiệu quả hơn Fully Connected layer cho xử lý ảnh. Trả lời bằng 2 khái niệm: parameter sharing và local connectivity.

3. **Batch Normalization:** Giải thích BatchNorm giải quyết vấn đề gì. Tại sao normalize theo batch mà không normalize theo feature? Công thức tính là gì?

4. **Attention Mechanism:** Giải thích cơ chế Scaled Dot-Product Attention. Tại sao chia cho $\sqrt{d_k}$? Viết công thức.

## Category 4 — Nâng cao (Advanced)

1. **Transformer Architecture:** So sánh Self-Attention vs Cross-Attention. Trong Transformer decoder, tại sao cần masked attention? Giải thích bằng ví dụ cụ thể.

2. **Residual Connections:** Chứng minh (bằng toán hoặc lập luận) tại sao residual connections giúp train được mạng rất sâu. Liên hệ với vanishing gradient problem.

3. **Regularization Theory:** So sánh L1 vs L2 regularization về mặt: (a) geometric interpretation, (b) sparsity, (c) khi nào nên dùng loại nào. Vẽ hoặc mô tả contour plot giải thích.

4. **Optimization Landscape:** Giải thích tại sao SGD thường generalize tốt hơn Adam. Liên hệ với sharpness of minima và flat minima hypothesis.

## Category 5 — Nghiên cứu (Research-Level)

1. **Scaling Laws:** Giải thích Chinchilla scaling laws. Cho compute budget cố định, nên phân bổ giữa model size và data size như thế nào? Tại sao?

2. **Diffusion Models:** Giải thích forward và reverse process trong DDPM. Viết loss function và giải thích tại sao training objective là dự đoán noise thay vì dự đoán $x_0$ trực tiếp.

3. **LoRA:** Giải thích cơ chế LoRA (Low-Rank Adaptation). Tại sao low-rank approximation hoạt động cho fine-tuning? Liên hệ với intrinsic dimensionality hypothesis.

4. **Architecture Design:** Nếu bạn cần thiết kế một model cho task X (pick: long-range sequence modeling), bạn sẽ chọn architecture nào và tại sao? So sánh ít nhất 3 options (Transformer, SSM/Mamba, RWKV).

## After Test: Output Format

```
## Kết quả Chẩn đoán

| Category | Điểm | Kết quả |
|----------|-------|---------|
| 1. Nền tảng Toán | X/4 | ✅/❌ |
| 2. ML Cơ bản | X/4 | ✅/❌ |
| 3. DL Core | X/4 | ✅/❌ |
| 4. Nâng cao | X/4 | ✅/❌ |
| 5. Nghiên cứu | X/4 | ✅/❌ |

**Trình độ chẩn đoán:** [Level]
**Điểm mạnh:** [...]
**Cần bổ sung:** [...]
**Đề xuất:** Đã cập nhật depth = [Level] trong student profile.
```
