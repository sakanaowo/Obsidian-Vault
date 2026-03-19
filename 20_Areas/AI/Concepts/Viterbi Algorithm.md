---
tags:
  - ai
  - nlp
  - algorithm
  - dynamic-programming
  - sequence-modeling
aliases:
  - Viterbi
  - Viterbi Decoding
status: evergreen
related:
  - "[[Hidden Markov Model]]"
  - "[[Dynamic Programming]]"
  - "[[Part-of-Speech Tagging]]"
  - "[[Beam Search]]"
---

# Viterbi Algorithm

> [!NOTE] ELI5
> Tưởng tượng bạn đang tìm đường đi ngắn nhất qua một mê cung có nhiều tầng. Tại mỗi tầng, có nhiều cửa để chọn, và mỗi cửa có chi phí khác nhau. **Thay vì thử tất cả các đường** (rất lâu!), Viterbi "thông minh" hơn: tại mỗi cửa ở tầng $k$, nó chỉ nhớ **đường tốt nhất để đến cửa đó** từ tầng 1. Cuối cùng, backtrack từ cửa tốt nhất ở tầng cuối để có đường tối ưu.

---

## 1. Bài Toán và Động Lực

### 1.1 Bài Toán Decoding

Cho [[Hidden Markov Model]] $\lambda = (A, B, \pi)$ và chuỗi quan sát $O = o_1, o_2, ..., o_T$, tìm chuỗi trạng thái ẩn **có xác suất cao nhất**:

$$Q^* = \argmax_{Q} P(Q | O, \lambda) = \argmax_{Q} P(O, Q | \lambda)$$

### 1.2 Tại Sao Không Duyệt Tất Cả?

Nếu có $N$ trạng thái và chuỗi dài $T$, số chuỗi khả dĩ là $N^T$. Ví dụ: $N=36$ tags, $T=20$ từ → $36^{20} \approx 10^{31}$ chuỗi. **Không khả thi!**

### 1.3 Nguyên Lý Tối Ưu (Bellman)

Viterbi dựa trên nhận xét quan trọng: **Đường đi tốt nhất đến trạng thái $(t, s)$ phải đi qua đường đi tốt nhất đến một trạng thái nào đó ở thời điểm $t-1$**.

Đây là **optimal substructure** — nền tảng của [[Dynamic Programming]].

---

## 2. Thuật Toán Chi Tiết

### 2.1 Định Nghĩa Biến Viterbi

$$v_s(t) = \max_{q_1,...,q_{t-1}} P(q_1, ..., q_{t-1}, q_t = s, o_1, ..., o_t | \lambda)$$

$v_s(t)$ là xác suất cao nhất của bất kỳ chuỗi trạng thái nào **kết thúc ở trạng thái $s$** tại thời điểm $t$, đã sinh ra quan sát $o_1, ..., o_t$.

### 2.2 Các Bước Thuật Toán

**Bước 1: Khởi tạo (Initialization)**

$$v_s(1) = \pi_s \cdot b_s(o_1), \quad \forall s \in Q$$

**Bước 2: Đệ quy (Recursion)**

Với $t = 2, 3, ..., T$:

$$v_s(t) = \max_{s' \in Q} \left[ v_{s'}(t-1) \cdot a_{s',s} \right] \cdot b_s(o_t)$$

$$bt_s(t) = argmax_{s' \in Q} \left[ v_{s'}(t-1) \cdot a_{s',s} \right]$$

Trong đó $bt_s(t)$ là **backpointer** — lưu trạng thái trước đó tốt nhất.

**Bước 3: Kết thúc (Termination)**

$$P^* = \max_{s \in Q} v_s(T)$$
$$q_T^* = argmax_{s \in Q} v_s(T)$$

**Bước 4: Backtracking**

$$q_t^* = bt_{q_{t+1}^*}(t+1), \quad t = T-1, T-2, ..., 1$$

### 2.3 Pseudocode

```python
def viterbi(observations, states, start_prob, trans_prob, emit_prob):
    T = len(observations)
    N = len(states)
    
    # Initialization
    V = [{} for _ in range(T)]
    backpointer = [{} for _ in range(T)]
    
    for s in states:
        V[0][s] = start_prob[s] * emit_prob[s][observations[0]]
        backpointer[0][s] = None
    
    # Recursion
    for t in range(1, T):
        for s in states:
            max_prob, best_prev = max(
                (V[t-1][s_prev] * trans_prob[s_prev][s], s_prev)
                for s_prev in states
            )
            V[t][s] = max_prob * emit_prob[s][observations[t]]
            backpointer[t][s] = best_prev
    
    # Termination
    best_last_state = max(V[T-1], key=V[T-1].get)
    best_prob = V[T-1][best_last_state]
    
    # Backtracking
    best_path = [best_last_state]
    for t in range(T-1, 0, -1):
        best_path.insert(0, backpointer[t][best_path[0]])
    
    return best_path, best_prob
```

---

## 3. Phân Tích Độ Phức Tạp

| Metric | Complexity |
|--------|------------|
| Thời gian | $O(N^2 \cdot T)$ |
| Không gian | $O(N \cdot T)$ |

So với brute force $O(N^T)$, đây là cải tiến **theo hàm mũ**.

**Giải thích:**
- Tại mỗi thời điểm $t$, tính $v_s(t)$ cho $N$ trạng thái
- Mỗi $v_s(t)$ cần xét $N$ trạng thái trước đó (để tìm max)
- Lặp qua $T$ thời điểm
- Tổng: $N \times N \times T = O(N^2 T)$

---

## 4. Xử Lý Numerical Underflow

### 4.1 Vấn Đề

Tích của nhiều xác suất nhỏ dẫn đến underflow (số quá nhỏ, trở thành 0 trong floating-point).

### 4.2 Giải Pháp: Log-space Viterbi

Làm việc với log-probabilities:

$$\log v_s(t) = \max_{s'} \left[ \log v_{s'}(t-1) + \log a_{s',s} \right] + \log b_s(o_t)$$

Phép nhân → phép cộng, phép max vẫn giữ nguyên.

---

## 5. Biến Thể và Mở Rộng

### 5.1 [[Beam Search]]

Thay vì giữ tất cả $N$ trạng thái tại mỗi bước, chỉ giữ top-$k$ (beam width). Trade-off: tốc độ vs optimality.

### 5.2 Higher-order Viterbi

Cho second-order HMM (trigram), trạng thái mở rộng thành cặp $(s_{t-1}, s_t)$. Độ phức tạp: $O(N^3 T)$.

### 5.3 A* Parsing

Kết hợp Viterbi với heuristic để prune không gian tìm kiếm trong parsing.

---

## 6. Ứng Dụng

- **[[Part-of-Speech Tagging]]:** Tìm chuỗi POS tag tối ưu cho câu
- **Speech Recognition:** Decode phoneme sequence từ acoustic features
- **Error Correction:** Viterbi decoder cho convolutional codes
- **Bioinformatics:** Gene prediction, sequence alignment

---

## 7. So Sánh Với Các Thuật Toán Liên Quan

| Algorithm | Output | Optimality | Complexity |
|-----------|--------|------------|------------|
| Viterbi | Best path | Exact | $O(N^2 T)$ |
| Forward | $P(O|\lambda)$ | Exact | $O(N^2 T)$ |
| Forward-Backward | $P(q_t=s|O)$ | Exact | $O(N^2 T)$ |
| Beam Search | Approximate best path | Approximate | $O(kNT)$ |

---

## TODO

- [ ] Thêm visualization của Viterbi lattice
- [ ] Code example với real POS tagging data
- [ ] So sánh với Forward algorithm (mục đích khác nhau)
