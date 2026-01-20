---
tags:
  - ai
  - nlp
  - machine-learning
  - probabilistic-model
  - sequence-modeling
  - hmm
aliases:
  - HMM
  - Hidden Markov Models
status: evergreen
related:
  - "[[Markov Chain]]"
  - "[[Viterbi Algorithm]]"
  - "[[Part-of-Speech Tagging]]"
  - "[[Conditional Random Fields (CRF)]]"
---

# Hidden Markov Model (HMM)

> [!NOTE] ELI5
> Tưởng tượng bạn đang nghe tiếng bước chân từ phòng bên cạnh. Bạn **không thể thấy** người đó đang làm gì (đi bộ? chạy? nhảy?), nhưng bạn **nghe được âm thanh** bước chân. Từ âm thanh này, bạn đoán được hành động. HMM hoạt động giống vậy: nó có những **trạng thái ẩn** (hành động thực sự) và những **quan sát được** (âm thanh), rồi sử dụng xác suất để suy ra trạng thái ẩn từ quan sát.

---

## 1. Định Nghĩa và Trực Giác

**Hidden Markov Model (HMM)** là một mô hình xác suất sinh (generative probabilistic model) cho dữ liệu **chuỗi** (sequential data), trong đó hệ thống được giả định chuyển đổi giữa các **trạng thái ẩn** (hidden/latent states) theo thời gian, và tại mỗi trạng thái, nó **sinh ra** (emit) một quan sát.

**Tại sao "Hidden"?** Vì ta không quan sát trực tiếp các trạng thái — ta chỉ thấy các output (observations). Nhiệm vụ của HMM là suy luận ngược từ observations về states.

**Tại sao "Markov"?** Vì trạng thái tiếp theo chỉ phụ thuộc vào trạng thái hiện tại (Markov property), không phụ thuộc vào lịch sử xa hơn.

---

## 2. Các Thành Phần Formal

Một HMM được định nghĩa bởi bộ 5 $\lambda = (Q, O, A, B, \pi)$:

### 2.1 Tập Trạng Thái Ẩn $Q$

$$Q = \{q_1, q_2, ..., q_N\}$$

$N$ trạng thái khả dĩ mà hệ thống có thể ở. Trong POS tagging, đây là các tags (NN, VB, DT, ...).

### 2.2 Tập Quan Sát $O$

$$O = \{o_1, o_2, ..., o_M\}$$

$M$ symbols quan sát khả dĩ. Trong POS tagging, đây là vocabulary (các từ).

### 2.3 Ma Trận Chuyển Trạng Thái $A$ (Transition Matrix)

$$A = [a_{ij}], \quad a_{ij} = P(q_j \text{ at } t+1 \mid q_i \text{ at } t)$$

$a_{ij}$ là xác suất chuyển từ trạng thái $i$ sang trạng thái $j$. Ràng buộc: $\sum_j a_{ij} = 1$ cho mọi $i$.

### 2.4 Ma Trận Phát Xạ $B$ (Emission Matrix)

$$B = [b_i(o_k)], \quad b_i(o_k) = P(o_k \mid q_i)$$

$b_i(o_k)$ là xác suất quan sát $o_k$ khi ở trạng thái $q_i$. Ràng buộc: $\sum_k b_i(o_k) = 1$ cho mọi $i$.

### 2.5 Phân Phối Khởi Tạo $\pi$

$$\pi = [\pi_i], \quad \pi_i = P(q_i \text{ at } t=1)$$

Xác suất bắt đầu ở mỗi trạng thái. Ràng buộc: $\sum_i \pi_i = 1$.

---

## 3. Ba Bài Toán Cơ Bản của HMM

### 3.1 Evaluation (Likelihood)

**Bài toán:** Cho HMM $\lambda$ và chuỗi quan sát $O = o_1, o_2, ..., o_T$, tính $P(O|\lambda)$.

**Giải pháp:** [[Forward Algorithm]] hoặc Backward Algorithm — sử dụng dynamic programming để tính hiệu quả trong $O(N^2 T)$ thay vì $O(N^T)$.

### 3.2 Decoding (State Inference)

**Bài toán:** Cho HMM $\lambda$ và chuỗi quan sát $O$, tìm chuỗi trạng thái ẩn $Q^* = \argmax_Q P(Q|O, \lambda)$.

**Giải pháp:** [[Viterbi Algorithm]] — dynamic programming tìm đường đi tốt nhất qua lattice.

### 3.3 Learning (Parameter Estimation)

**Bài toán:** Cho tập dữ liệu quan sát, học tham số $\lambda = (A, B, \pi)$.

**Giải pháp:**
- **Supervised:** Nếu có labels, dùng Maximum Likelihood Estimation (đếm và normalize)
- **Unsupervised:** Nếu không có labels, dùng [[Baum-Welch Algorithm]] (EM algorithm cho HMM)

---

## 4. Các Giả Định Quan Trọng

HMM hoạt động dưới hai giả định mạnh:

### 4.1 Markov Assumption (First-order)

$$P(q_t | q_1, ..., q_{t-1}) = P(q_t | q_{t-1})$$

Trạng thái hiện tại chỉ phụ thuộc vào trạng thái ngay trước đó.

### 4.2 Output Independence

$$P(o_t | q_1, ..., q_t, o_1, ..., o_{t-1}) = P(o_t | q_t)$$

Quan sát tại thời điểm $t$ chỉ phụ thuộc vào trạng thái tại $t$, không phụ thuộc vào các quan sát/trạng thái khác.

> [!WARNING] Hạn Chế Của Giả Định
> Trong thực tế, các giả định này thường bị vi phạm. Ví dụ trong ngôn ngữ, từ tiếp theo phụ thuộc mạnh vào nhiều từ trước đó (long-range dependencies). Đây là lý do các mô hình như [[Conditional Random Fields (CRF)]], LSTM, và [[Transformer]] thường outperform HMM cho nhiều tác vụ.

---

## 5. Ứng Dụng

### 5.1 Trong NLP
- **[[Part-of-Speech Tagging]]:** Tag ẩn, từ quan sát
- **Named Entity Recognition:** Entity type ẩn, từ quan sát
- **Speech Recognition:** Phoneme ẩn, acoustic features quan sát

### 5.2 Ngoài NLP
- **Bioinformatics:** Gene finding, protein structure prediction
- **Finance:** Regime detection trong time series
- **Robotics:** Localization và SLAM

---

## 6. HMM vs Discriminative Models

| Aspect | HMM (Generative) | CRF (Discriminative) |
|--------|------------------|---------------------|
| Mô hình | $P(O, Q)$ | $P(Q|O)$ |
| Independence assumptions | Mạnh (output independence) | Yếu hơn (có thể dùng overlapping features) |
| Feature engineering | Hạn chế | Linh hoạt |
| Handling unknown words | Khó | Dễ hơn với features |
| Training | Nhanh (closed-form MLE) | Chậm hơn (gradient descent) |

---

## TODO

- [ ] Viết concept note chi tiết cho [[Forward Algorithm]] và [[Baum-Welch Algorithm]]
- [ ] So sánh với MEMM (Maximum Entropy Markov Model) và label bias problem
- [ ] Thêm code example với hmmlearn hoặc pomegranate
