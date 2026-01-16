---
type: concept
title: Autoencoders
aliases:
  - Autoencoder
tags:
  - ai
  - representation-learning
  - unsupervised-learning
---

**Autoencoder** là một họ mô hình học biểu diễn trong đó ta học một ánh xạ mã hóa $f_\theta$ (encoder) biến đầu vào $x$ thành biểu diễn ẩn $z=f_\theta(x)$ và một ánh xạ giải mã $g_\phi$ (decoder) tái tạo $\hat{x}=g_\phi(z)$ sao cho $\hat{x}$ “gần” $x$ theo một hàm mất mát. Điểm cốt lõi không nằm ở việc tái tạo cho giống “hình thức”, mà nằm ở chỗ **để tái tạo được, mô hình buộc phải nén**: $z$ phải giữ những yếu tố của $x$ mà decoder cần, trong khi loại bỏ những phần “không cần thiết” theo thiên kiến kiến trúc và mục tiêu tối ưu.

Trong cách nhìn “first principles”, autoencoder đang giải một bài toán tối ưu hóa dạng $$\min_{\theta,\phi}\;\mathbb{E}_{x\sim\mathcal{D}}\big[\mathcal{L}(x, g_\phi(f_\theta(x)))\big].$$ Nếu $z$ có chiều rất lớn và decoder đủ mạnh, bài toán có thể suy biến thành sao chép (identity). Vì vậy, các biến thể autoencoder luôn đưa vào một “nút thắt” (bottleneck) hoặc một dạng **corruption/regularization** để ép representation hữu ích, ví dụ giảm chiều, phạt độ phức tạp, hoặc làm hỏng đầu vào rồi bắt mô hình khôi phục.

**Denoising Autoencoder (DAE)** làm hỏng đầu vào theo một toán tử nhiễu $\tilde{x}=q(x)$ rồi học khôi phục $x$ từ $\tilde{x}$. Khi $q$ là “che một phần tín hiệu”, DAE trở thành nền tảng khái niệm của [[Masked Autoencoders (MAE)]]. Sự khác biệt quan trọng là mức độ che và kiến trúc: với thị giác, nếu che quá ít, mô hình có thể phục hồi bằng tương quan cục bộ; nếu che đủ nhiều, mô hình bị buộc phải học cấu trúc toàn cục và quan hệ phần–toàn thể.

Ví dụ tối giản cho ảnh dạng patch: ta che ngẫu nhiên $r$ phần trăm patch, encoder chỉ thấy phần còn lại, decoder dự đoán patch bị che và loss thường là MSE trên pixel. Cơ chế này nối trực tiếp sang [[Self-Supervised Learning (Computer Vision)]], vì ta không cần nhãn; “nhãn” được tạo từ chính dữ liệu gốc thông qua phép che.

