---
tags:
  - ai
  - machine-learning
  - probabilistic-model
  - stochastic-process
aliases:
  - Markov Chains
  - Markov Process
  - Discrete-time Markov Chain
  - DTMC
status: evergreen
related:
  - "[[Hidden Markov Model]]"
  - "[[N-gram Language Model]]"
  - "[[Probability Theory]]"
---

# Markov Chain

> [!NOTE] ELI5
> Tưởng tượng bạn đang chơi cờ cá ngựa. Vị trí tiếp theo của bạn **chỉ phụ thuộc vào vị trí hiện tại** và con xúc xắc — không quan trọng bạn đã đi qua ô nào trước đó. Markov Chain mô tả kiểu hệ thống như vậy: **tương lai chỉ phụ thuộc vào hiện tại, không phụ thuộc vào quá khứ**. Đây gọi là tính chất "không nhớ" (memoryless).

---

## 1. Định Nghĩa và Markov Property

### 1.1 Định Nghĩa Informal

**Markov Chain** là một mô hình xác suất mô tả chuỗi các trạng thái ngẫu nhiên, trong đó xác suất chuyển sang trạng thái tiếp theo **chỉ phụ thuộc vào trạng thái hiện tại**.

### 1.2 Markov Property (Memorylessness)

Formally, cho chuỗi biến ngẫu nhiên $X_1, X_2, ..., X_t, ...$:

$$P(X_{t+1} = s_{t+1} | X_t = s_t, X_{t-1} = s_{t-1}, ..., X_1 = s_1) = P(X_{t+1} = s_{t+1} | X_t = s_t)$$

**Diễn giải:** "Biết hiện tại, tương lai độc lập với quá khứ."

### 1.3 Tại Sao Giả Định Này Hữu Ích?

- **Đơn giản hóa tính toán:** Không cần lưu trữ/xét toàn bộ lịch sử
- **Đủ mạnh cho nhiều ứng dụng:** Nhiều hệ thống thực tế xấp xỉ Markov
- **Nền tảng cho các mô hình phức tạp hơn:** HMM, MDP, MCMC

---

## 2. Các Thành Phần Formal

Một Markov Chain được định nghĩa bởi:

### 2.1 Tập Trạng Thái $S$

$$S = \{s_1, s_2, ..., s_N\}$$

Tập hữu hạn (hoặc đếm được) các trạng thái khả dĩ.

### 2.2 Ma Trận Chuyển Trạng Thái $P$ (Transition Matrix)

$$P = [p_{ij}], \quad p_{ij} = P(X_{t+1} = s_j | X_t = s_i)$$

Ma trận $N \times N$ với các tính chất:
- $p_{ij} \geq 0$ (non-negative)
- $\sum_j p_{ij} = 1$ (hàng tổng = 1, stochastic matrix)

**Ví dụ (Thời tiết):**

|  | Sunny | Rainy |
|--|-------|-------|
| Sunny | 0.8 | 0.2 |
| Rainy | 0.4 | 0.6 |

### 2.3 Phân Phối Khởi Tạo $\pi^{(0)}$

$$\pi^{(0)} = [\pi_1^{(0)}, \pi_2^{(0)}, ..., \pi_N^{(0)}], \quad \pi_i^{(0)} = P(X_1 = s_i)$$

---

## 3. Tính Toán Với Markov Chain

### 3.1 Xác Suất Ở Trạng Thái Sau $n$ Bước

$$\pi^{(n)} = \pi^{(0)} \cdot P^n$$

Ma trận $P^n$ cho ta xác suất chuyển sau $n$ bước.

### 3.2 Xác Suất Của Một Chuỗi Cụ Thể

$$P(X_1 = s_{i_1}, X_2 = s_{i_2}, ..., X_T = s_{i_T}) = \pi_{i_1}^{(0)} \cdot p_{i_1 i_2} \cdot p_{i_2 i_3} \cdots p_{i_{T-1} i_T}$$

---

## 4. Phân Loại Trạng Thái

### 4.1 Absorbing State

Trạng thái $s_i$ là **absorbing** nếu $p_{ii} = 1$ (một khi vào, không thể ra).

### 4.2 Transient vs Recurrent

- **Transient:** Có xác suất >0 không bao giờ quay lại
- **Recurrent:** Chắc chắn sẽ quay lại (xác suất = 1)

### 4.3 Periodic vs Aperiodic

- **Periodic (chu kỳ $d$):** Chỉ có thể quay lại sau $d, 2d, 3d, ...$ bước
- **Aperiodic:** Có thể quay lại sau bất kỳ số bước nào đủ lớn

---

## 5. Phân Phối Dừng (Stationary Distribution)

### 5.1 Định Nghĩa

Phân phối $\pi^*$ là **stationary** nếu:

$$\pi^* = \pi^* \cdot P$$

Tức là: nếu bắt đầu từ $\pi^*$, sau mỗi bước vẫn ở $\pi^*$.

### 5.2 Định Lý Ergodic

Nếu Markov Chain là:
- **Irreducible:** Mọi trạng thái đều đến được từ mọi trạng thái khác
- **Aperiodic:** Không có chu kỳ

Thì tồn tại **unique stationary distribution** $\pi^*$, và:

$$\lim_{n \to \infty} P^n = \mathbf{1} \cdot \pi^*$$

(Mọi hàng của $P^n$ hội tụ về $\pi^*$)

---

## 6. Ứng Dụng

### 6.1 Trong NLP

- **[[N-gram Language Model]]:** Xác suất từ tiếp theo dựa trên $n-1$ từ trước
- **[[Hidden Markov Model]]:** Markov chain trên trạng thái ẩn
- **Text generation:** Markov text generators

### 6.2 Trong Machine Learning

- **MCMC (Markov Chain Monte Carlo):** Sampling từ phân phối phức tạp
- **PageRank:** Random walk trên đồ thị web

### 6.3 Trong Các Lĩnh Vực Khác

- **Finance:** Mô hình giá cổ phiếu, credit rating transitions
- **Biology:** Mô hình tiến hóa DNA
- **Physics:** Mô hình hạt trong không gian

---

## 7. Markov Chain Bậc Cao (Higher-order)

### 7.1 Second-order Markov Chain

$$P(X_{t+1} | X_t, X_{t-1}, ...) = P(X_{t+1} | X_t, X_{t-1})$$

Phụ thuộc vào 2 trạng thái gần nhất.

### 7.2 Biến Đổi Về First-order

Có thể biến second-order thành first-order bằng cách mở rộng không gian trạng thái: trạng thái mới là **cặp** $(X_{t-1}, X_t)$.

---

## TODO

- [ ] Thêm ví dụ minh họa với transition diagram
- [ ] Code example tính stationary distribution
- [ ] Liên kết với [[Markov Decision Process (MDP)]] trong Reinforcement Learning
