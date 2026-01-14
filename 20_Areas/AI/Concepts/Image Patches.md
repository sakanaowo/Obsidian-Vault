---
type: concept
title: Image Patches
aliases:
  - Patch (Vision)
  - Patches
tags:
  - ai
  - computer-vision
  - transformers
---

Trong các mô hình kiểu [[Vision Transformers (ViT)]], **image patch** là một cách biểu diễn ảnh bằng cách chia ảnh $H\\times W\\times C$ thành các khối nhỏ không chồng lấp kích thước $P\\times P\\times C$. Mỗi patch được “trải phẳng” (flatten) thành một vector và đi qua một lớp tuyến tính để trở thành một **token embedding**. Khi đó, ảnh được xem như một chuỗi token có độ dài $N = (H/P)\\cdot(W/P)$, và mô hình Transformer có thể áp dụng trực tiếp cơ chế self-attention lên chuỗi token này.

Điểm quan trọng của patchification là nó tạo ra một “đơn vị thao tác” trung gian giữa pixel và đối tượng. Pixel quá nhỏ và dư thừa; đối tượng quá cao tầng và cần nhãn. Patch nằm ở giữa: đủ nhỏ để tạo bài toán tự giám sát dựa trên cấu trúc cục bộ/toàn cục, nhưng đủ lớn để giảm chiều dài chuỗi so với làm việc trực tiếp trên pixel (như iGPT). Trong [[Masked Autoencoders (MAE)]], patch chính là đơn vị bị che: ta loại bỏ một tỉ lệ lớn patch khỏi encoder để giảm compute, và yêu cầu decoder dự đoán pixel của patch bị che.

Một nuance kỹ thuật: self-attention có chi phí gần bậc hai theo $N$, nên lựa chọn $P$ (kích thước patch) là trade-off giữa “độ chi tiết” và “chi phí”. Patch nhỏ (P nhỏ) cho biểu diễn chi tiết nhưng $N$ lớn và tốn compute; patch lớn giảm compute nhưng tăng nguy cơ mất cấu trúc tinh.

