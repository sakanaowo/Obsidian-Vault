---
type: concept
title: Self-Supervised Learning (Computer Vision)
aliases:
  - Self-supervised learning (vision)
  - SSL (Computer Vision)
tags:
  - ai
  - computer-vision
  - representation-learning
---

**Self-supervised learning** trong thị giác máy tính là cách học representation từ dữ liệu không nhãn bằng cách tạo ra một “nhiệm vụ giả” (pretext task) mà nhãn có thể được sinh ra từ chính dữ liệu. Điểm không tầm thường của SSL là: ta không quan tâm tối đa hóa điểm số của pretext task, mà quan tâm pretext task đó có buộc mô hình học ra các bất biến/khái niệm hữu ích cho downstream hay không.

Có hai dòng tư duy lớn. Dòng **contrastive** xây dựng loss dựa trên việc kéo gần các “view” khác nhau của cùng một ảnh và đẩy xa ảnh khác; sức mạnh của nó phụ thuộc nhiều vào **data augmentation** (vì augmentation định nghĩa “cái gì là cùng một ảnh”). Dòng **reconstruction/generative** (như [[Masked Autoencoders (MAE)]]) làm hỏng một phần tín hiệu rồi yêu cầu mô hình tái tạo lại; sức mạnh của nó phụ thuộc nhiều vào việc corruption có triệt tiêu được shortcut hay không (ví dụ masking ratio đủ cao để tránh nội suy cục bộ).

Trong thực hành, SSL được đánh giá bằng hai kiểu phổ biến. **Linear probing** cố định backbone và huấn luyện một đầu phân loại tuyến tính để đo “mức tuyến tính hóa” của representation; còn **fine-tuning** cập nhật toàn bộ mô hình theo downstream objective. Hai thước đo này có thể cho xu hướng khác nhau, vì linear probing đòi hỏi representation đã “phù hợp” ngay, còn fine-tuning cho phép mô hình điều chỉnh sâu.

