---
type: concept
title: Vision Transformers (ViT)
aliases:
  - ViT
  - Vision Transformer
tags:
  - ai
  - computer-vision
  - transformers
---

**Vision Transformers (ViT)** là một cách áp dụng kiến trúc Transformer vào thị giác bằng cách biến ảnh thành một chuỗi token thông qua [[Image Patches]]. Mỗi patch được nhúng bằng một phép chiếu tuyến tính, cộng thêm **positional embedding**, rồi đi qua các block gồm **self-attention** và MLP giống Transformer trong NLP. Trực giác: self-attention cho phép mô hình “trộn thông tin” giữa các vùng ảnh ở khoảng cách xa mà không cần inductive bias cục bộ mạnh như CNN.

Về mặt cơ chế, ViT chuyển bài toán thị giác sang bài toán chuỗi: độ dài chuỗi là số patch $N$. Đây vừa là ưu thế (tận dụng Transformer) vừa là nút thắt compute (attention bậc hai theo $N$). Vì vậy, nhiều phương pháp tự giám sát hiện đại hoặc thay đổi cách tạo token (giảm $N$), hoặc thay đổi cách attention vận hành, hoặc thay đổi pipeline pretrain để encoder không phải nhìn toàn bộ token mọi lúc.

[[Masked Autoencoders (MAE)]] là một ví dụ đặc biệt “tôn trọng” ViT: encoder MAE hầu như là ViT vanilla, nhưng chỉ chạy trên subset patch nhìn thấy. Điều này giữ nguyên ưu điểm kiến trúc của ViT trong downstream (encoder nhìn ảnh đầy đủ), đồng thời tận dụng giảm compute trong pretrain (encoder nhìn ảnh thưa).

