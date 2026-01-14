---
tags:
  - Math/InformationTheory
  - AI/LossFunction
  - Concept
alias:
  - Kullback-Leibler Divergence
  - Relative Entropy
created: 2026-01-04
---

### Định nghĩa

**KL Divergence** (Kullback-Leibler divergence), hay còn gọi là **Relative Entropy**, là một thước đo sự **khác biệt** giữa hai phân phối xác suất $P$ và $Q$.

Trong Machine Learning:
*   $P$: Thường là phân phối dữ liệu thật (Ground truth / True distribution).
*   $Q$: Là phân phối do mô hình dự đoán (Predicted / Model distribution).
*   Mục tiêu: Kéo $Q$ càng gần $P$ càng tốt $\rightarrow$ Giảm thiểu KL Divergence.

### Công thức Toán học

$$D_{KL}(P || Q) = \sum_{i} P(x_i) \cdot \log \frac{P(x_i)}{Q(x_i)}$$

Hoặc viết dưới dạng hiệu của hai giá trị Log:

$$D_{KL}(P || Q) = \sum_{i} P(x_i) \log P(x_i) - \sum_{i} P(x_i) \log Q(x_i)$$

### Tính chất quan trọng: Bất đối xứng (Asymmetry)

Khác với khoảng cách hình học (Distance) thông thường (như Euclidean), KL Divergence **không đối xứng**:

$$D_{KL}(P || Q) \neq D_{KL}(Q || P)$$

*   *Ý nghĩa:* Việc dùng $Q$ để xấp xỉ $P$ khác với việc dùng $P$ để xấp xỉ $Q$.

### Liên hệ với Cross-Entropy

Trong huấn luyện Neural Networks, ta thường tối ưu hóa **Cross-Entropy** thay vì trực tiếp KL Divergence, vì chúng có mối quan hệ chặt chẽ:

$$H(P, Q) = H(P) + D_{KL}(P || Q)$$

*   **Cross-Entropy** = **Entropy của P** (Hằng số đối với dữ liệu huấn luyện) + **KL Divergence**.
*   Do đó, **Minimizing Cross-Entropy $\equiv$ Minimizing KL Divergence**.

### Ví dụ trực quan

Tưởng tượng $P$ là bản đồ địa hình thực tế, $Q$ là bản đồ bạn vẽ.
*   $D_{KL}(P||Q)$: "Độ sai lệch" khi bạn dùng bản đồ vẽ để đi thực địa. Nếu bản đồ vẽ sai chỗ quan trọng (nơi $P$ cao mà $Q$ thấp), hậu quả sẽ rất lớn (giá trị KL cao).
