---
title: "Maximum Likelihood Estimation"
aliases: [MLE, ước lượng hợp lý tối đa]
tags: [concept, statistics, machine-learning]
created: 2026-03-16
---

# Maximum Likelihood Estimation

> [!NOTE] ELI5
> Hãy tưởng tượng mô hình là một "máy kể chuyện" về dữ liệu. MLE chọn tham số để câu chuyện đó làm dữ liệu bạn thấy ngoài đời trở nên "hợp lý" nhất.
> Nếu một bộ tham số làm dữ liệu thật xuất hiện với xác suất cao, ta tin bộ tham số đó tốt hơn.
> Nói ngắn gọn: MLE tìm tham số khiến dữ liệu quan sát được trở thành kịch bản dễ xảy ra nhất.

## 1) Bản chất: Vì sao phải là "likelihood lớn nhất"?

**Claim:** MLE hợp lý vì học từ dữ liệu nên tiêu chí tự nhiên nhất là: "tham số nào giải thích dữ liệu đã quan sát tốt nhất".

**Reasoning:** Trong supervised learning, ta có dữ liệu cố định và tham số chưa biết. Vì dữ liệu đã xảy ra, ta đánh giá mỗi bộ tham số bằng mức độ nó gán xác suất cho chính dữ liệu đó. Bộ nào gán xác suất cao hơn thì "đỡ bất ngờ" hơn.

**Evidence (công thức):**

Với dữ liệu độc lập có điều kiện $(x_i, y_i)$:

$$
\hat{\theta}_{\text{MLE}}
=\arg\max_{\theta}\prod_{i=1}^{n} p(y_i\mid x_i;\theta)
$$

Vì tích nhiều số nhỏ dễ gây underflow, ta lấy log và đổi max thành min:

$$
\hat{\theta}_{\text{MLE}}
=\arg\min_{\theta}\left[-\sum_{i=1}^{n}\log p(y_i\mid x_i;\theta)\right]
$$

Biểu thức trong ngoặc là **negative log-likelihood (NLL)**.

## 2) Ví dụ đời thường và trực giác

Bạn tung đồng xu 10 lần, thấy 8 ngửa 2 sấp. Có 2 giả thuyết:

- Đồng xu công bằng: $p(\text{ngửa})=0.5$
- Đồng xu lệch: $p(\text{ngửa})=0.8$

Dữ liệu "8 ngửa, 2 sấp" sẽ có likelihood cao hơn dưới giả thuyết $0.8$ so với $0.5$. Vì vậy MLE nghiêng về $p\approx0.8$.

Ý nghĩa: MLE không hỏi "tham số đúng tuyệt đối là gì", mà hỏi "tham số nào làm dữ liệu hiện tại bớt bất thường nhất".

## 3) Công thức và giải thích ký hiệu

$$
\mathcal{L}(\theta)=\prod_{i=1}^{n} p(y_i\mid x_i;\theta)
$$

- $\theta$: tham số mô hình (weights, bias, ...)
- $n$: số mẫu dữ liệu
- $p(y_i\mid x_i;\theta)$: xác suất mô hình gán cho nhãn đúng $y_i$ khi biết đầu vào $x_i$
- $\mathcal{L}(\theta)$: likelihood tổng của cả tập dữ liệu

Sau khi log:

$$
\ell(\theta)=\log \mathcal{L}(\theta)=\sum_{i=1}^{n}\log p(y_i\mid x_i;\theta)
$$

Tối ưu thực tế:

$$
\min_{\theta}\;\text{NLL}(\theta)
=\min_{\theta}\left[-\ell(\theta)\right]
$$

## 4) Hai trường hợp quan trọng trong Deep Learning

### 4.1 Gaussian likelihood  ->  MSE (Regression)

Giả sử:

$$
y_i=f_\theta(x_i)+\varepsilon_i,\quad \varepsilon_i\sim\mathcal{N}(0,\sigma^2)
$$

Khi đó:

$$
p(y_i\mid x_i;\theta)
=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}\right)
$$

Lấy NLL và bỏ hằng số không phụ thuộc $\theta$:

$$
\min_{\theta}\sum_{i=1}^{n}(y_i-f_\theta(x_i))^2
$$

Tức là MLE dưới nhiễu Gaussian dẫn tới **MSE loss**.

### 4.2 Categorical likelihood  ->  Cross-Entropy (Classification)

Với phân loại đa lớp, mô hình cho xác suất $\hat{\mathbf{y}}_i$ (thường từ softmax). Nhãn thật one-hot là $\mathbf{y}_i$.

$$
	ext{NLL}
=-\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik}\log \hat{y}_{ik}
$$

Đây chính là **cross-entropy loss** dùng trong phân loại.

## 5) Ưu điểm, giới hạn và điều kiện

### Ưu điểm

- Tiêu chí xác suất rõ ràng, có diễn giải thống kê chặt.
- Dẫn đến các loss quen thuộc (MSE, cross-entropy).
- Hợp với tối ưu gradient trong mô hình lớn.

### Giới hạn

- Nhạy với giả định phân phối (sai giả định -> ước lượng kém).
- Dễ bị outlier ảnh hưởng mạnh (đặc biệt với Gaussian/MSE).
- Có thể overfit nếu mô hình quá linh hoạt và dữ liệu ít.

### Điều kiện quan trọng

- Dữ liệu thường giả định i.i.d. hoặc độc lập có điều kiện.
- Mô hình phải đủ biểu diễn để xấp xỉ phân phối thật.

## 6) Bias-Variance và tính nhất quán (consistency)

- **Bias:** độ lệch trung bình của estimator so với tham số thật.
- **Variance:** độ dao động estimator khi thay mẫu dữ liệu.
- **Trade-off:** mô hình phức tạp giảm bias nhưng có thể tăng variance.

Với điều kiện regularity chuẩn và mô hình đúng dạng, MLE có tính chất tốt khi $n$ lớn:

- **Consistency:** $\hat{\theta}_{\text{MLE}} \xrightarrow[]{p} \theta^*$
- **Asymptotic normality:** gần chuẩn quanh $\theta^*$ khi $n$ lớn
- **Asymptotic efficiency:** đạt cận Cramer-Rao trong lớp estimator không chệch (tiệm cận)

## 7) MLE và MAP: khác nhau ở đâu?

- **MLE:** chỉ dùng dữ liệu, tối đa hóa $p(D\mid\theta)$.
- **MAP:** thêm prior, tối đa hóa $p(\theta\mid D)\propto p(D\mid\theta)p(\theta)$.

Hệ quả quan trọng trong DL:

- Gaussian prior trên tham số tương đương thêm regularization kiểu $L_2$.
- Nói cách khác, nhiều kỹ thuật regularization có thể nhìn như Bayesian prior.

## 8) Checklist thực hành cho training

1. Chọn likelihood theo bản chất bài toán (Gaussian, Bernoulli, Categorical...).
2. Viết NLL tương ứng, kiểm tra có khớp loss trong framework không.
3. Xác nhận đầu vào loss đúng dạng (logits hay probabilities).
4. Theo dõi train/val gap để tránh tối ưu likelihood nhưng mất generalization.

## Tóm tắt một dòng

**MLE = chọn tham số làm dữ liệu quan sát được có khả năng xuất hiện cao nhất; trong DL điều này sinh ra trực tiếp các loss chuẩn như MSE và cross-entropy.**

## TODO

- [ ] Bổ sung ví dụ Bernoulli chi tiết với bài toán spam/ham và log-odds.
- [ ] Thêm mục "MLE under model misspecification" (mô hình sai dạng thì tối ưu cái gì).
- [ ] Viết note riêng về MAP estimation và Bayesian inference rồi liên kết qua lại.
