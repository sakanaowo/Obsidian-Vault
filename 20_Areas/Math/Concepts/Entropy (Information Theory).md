---
tags:
  - Math/InformationTheory
  - AI/Foundation
  - Concept
alias:
  - Shannon Entropy
created: 2026-01-04
---

### Định nghĩa (Intuition First)

**Entropy** (Ký hiệu: $H$) là thước đo mức độ **"ngạc nhiên" (surprise)** hoặc **"bất định" (uncertainty)** của một biến ngẫu nhiên.
-   Nếu một sự kiện chắc chắn xảy ra 100% (ví dụ: Mặt trời mọc đằng Đông), Entropy = 0 (Không có gì ngạc nhiên).
-   Nếu một sự kiện hoàn toàn ngẫu nhiên và khó đoán (ví dụ: Tung đồng xu công bằng), Entropy đạt cực đại.

Trong Khoa học máy tính, Entropy đo lường **số lượng bits trung bình tối thiểu** cần thiết để mã hóa thông tin của sự kiện đó.

### Công thức Toán học

Với một biến ngẫu nhiên rời rạc $x$ có phân phối xác suất $P(x)$, Entropy được tính bằng:

$$H(x) = - \sum_{i=1}^{n} P(x_i) \cdot \log_b P(x_i)$$

*   $P(x_i)$: Xác suất xảy ra sự kiện thứ $i$.
*   $\log_b$: Thường dùng log cơ số 2 (đơn vị là bits), cơ số $e$ (nats), hoặc 10 (bans).
*   Dấu âm ($-$) ở đầu để đảm bảo kết quả dương (vì $\log(p) \leq 0$ khi $0 \leq p \leq 1$). 

### Ví dụ minh họa: Tung đồng xu

1.  **Đồng xu công bằng (Fair coin):**
    *   Head (Ngửa): $P=0.5$
    *   Tail (Sấp): $P=0.5$
    *   $H(x) = -(0.5 \log_2 0.5 + 0.5 \log_2 0.5) = -(-0.5 - 0.5) = \mathbf{1 \text{ bit}}$.
    *   *Ý nghĩa:* Cần 1 bit để biết kết quả (0 hoặc 1). Sự bất định cao nhất.

2.  **Đồng xu bị lệch (Biased coin):**
    *   Head: $P=0.9$ (Rất dễ đoán ra ngửa)
    *   Tail: $P=0.1$
    *   $H(x) = -(0.9 \log_2 0.9 + 0.1 \log_2 0.1) \approx \mathbf{0.469 \text{ bits}}$.
    *   *Ý nghĩa:* Entropy thấp hơn vì ta "ít ngạc nhiên" hơn (đa phần là ngửa).

### Vai trò trong AI/NLP

Trong NLP, chúng ta muốn mô hình ngôn ngữ (Language Model) có **Cross-Entropy thấp** trên tập dữ liệu kiểm thử. Điều này có nghĩa là mô hình không bị "ngạc nhiên" khi nhìn thấy từ tiếp theo, tức là nó đã dự đoán chính xác.
