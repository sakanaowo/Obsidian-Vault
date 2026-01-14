---
type: concept
title: Linear Probing
aliases:
  - Linear probe
tags:
  - ai
  - representation-learning
---

**Linear probing** là một giao thức đánh giá chất lượng representation trong học tự giám sát: ta đóng băng (freeze) backbone đã pretrain, rồi huấn luyện một lớp tuyến tính (ví dụ softmax linear classifier) trên downstream labels. Nếu một lớp tuyến tính đã đủ để tách lớp tốt, ta nói representation có mức “tuyến tính hóa” cao đối với tác vụ đó.

Điểm quan trọng là linear probing không cho phép backbone thích nghi; vì vậy nó nhạy với việc backbone có “đúng loại thông tin” ở đúng tầng hay không. Trong [[Masked Autoencoders (MAE)]], paper cho thấy thiết kế decoder (độ sâu/độ rộng) ảnh hưởng mạnh tới linear probing hơn so với fine-tuning. Cách hiểu cơ chế là: nếu decoder quá yếu, encoder phải mang nhiều chi tiết phục vụ tái tạo pixel, làm representation cuối khó tuyến tính hóa cho nhận dạng. Decoder đủ mạnh có thể hấp thụ phần “chuyên biệt reconstruction”, để encoder giữ latent trừu tượng hơn.

Vì vậy, khi đọc kết quả SSL, linear probing và fine-tuning là hai phép đo bổ sung chứ không thay thế nhau: linear probing đo “tính sẵn sàng” của representation, còn fine-tuning đo “khả năng chuyển hóa” khi được phép cập nhật tham số.

