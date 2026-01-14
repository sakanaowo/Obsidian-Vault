---
type: concept
title: Masking Ratio
aliases:
  - Mask ratio
tags:
  - ai
  - self-supervised-learning
---

**Masking ratio** (tỉ lệ che) là phần trăm của tín hiệu đầu vào bị loại bỏ/che đi trong các bài toán tự giám sát kiểu masked modeling. Nếu ảnh được chia thành $N$ patch, masking ratio $r$ nghĩa là có $rN$ patch bị che và chỉ còn $(1-r)N$ patch “nhìn thấy”. Trong [[Masked Autoencoders (MAE)]], giá trị $r$ điển hình là rất cao (khoảng 0.75), trái ngược với BERT (khoảng 0.15) trong NLP.

Vì sao tỉ lệ che tối ưu của thị giác lại cao? Lý do nằm ở **mật độ thông tin** và tính dư thừa. Pixel/patch có tương quan không gian rất mạnh; nếu chỉ che ít, mô hình có thể đoán phần bị che bằng cách nội suy cục bộ mà không cần hiểu đối tượng. Tăng $r$ làm giảm khả năng “ăn gian”, biến việc tái tạo thành một dạng suy luận toàn cục: mô hình phải dựa vào cấu trúc còn lại để quyết định phần thiếu trông như thế nào ở mức hợp nghĩa. Khi $r$ quá cao (ví dụ 0.95), tín hiệu còn lại quá ít có thể làm bài toán trở nên mơ hồ; nhưng một mức “cao vừa đủ” tạo ra áp lực học representation tốt.

Masking ratio còn quyết định compute. Với backbone Transformer, self-attention có chi phí bậc hai theo số token, nên giảm token đầu vào encoder bằng cách tăng $r$ có thể giảm compute rất mạnh. Tuy nhiên, nếu decoder vẫn phải xử lý full token để tái tạo, tổng compute không giảm theo $(1-r)^2$ một cách thuần túy; vì vậy MAE kết hợp masking ratio cao với decoder nhẹ để đạt lợi ích thực tế.

