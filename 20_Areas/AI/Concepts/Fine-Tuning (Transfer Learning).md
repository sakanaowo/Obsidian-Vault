---
type: concept
title: Fine-Tuning (Transfer Learning)
aliases:
  - Fine-tuning
  - Finetuning
tags:
  - ai
  - transfer-learning
---

**Fine-tuning** (trong bối cảnh **transfer learning**) là quá trình lấy một mô hình đã được huấn luyện trước (pre-trained) và tiếp tục tối ưu hóa nó trên một tác vụ/miền dữ liệu mới. Điểm phân biệt với [[Supervised Fine-Tuning (SFT)]] trong NLP là: “fine-tuning” là khái niệm tổng quát (có thể có giám sát hoặc tự giám sát), còn SFT là một trường hợp cụ thể cho LLM với dữ liệu instruction/QA. Trong thị giác, fine-tuning thường nghĩa là: giữ kiến trúc backbone (ví dụ [[Vision Transformers (ViT)]]) và tối ưu end-to-end trên loss của downstream task (classification/detection/segmentation).

Về cơ chế, fine-tuning giải quyết một bài toán tối ưu hóa có khởi tạo tốt: thay vì bắt đầu từ tham số ngẫu nhiên $\\theta_0$, ta bắt đầu từ $\\theta_{pre}$ đã “mã hóa” các đặc trưng phổ quát. Nếu pre-training objective đã buộc mô hình học cấu trúc hữu ích (ví dụ masked reconstruction của [[Masked Autoencoders (MAE)]]), thì $\\theta_{pre}$ nằm gần một nghiệm tốt cho downstream, giúp (i) hội tụ nhanh hơn, (ii) cần ít dữ liệu nhãn hơn, và (iii) tổng quát tốt hơn khi mô hình rất lớn.

Một nuance quan trọng là “mismatch giữa pretrain và downstream”. Nếu pretrain làm mô hình quen với một phân phối đầu vào khác với downstream (ví dụ encoder thấy nhiều **mask token** trong pretrain nhưng downstream lại thấy ảnh đầy đủ), fine-tuning có thể phải “sửa” đại diện quá nhiều và mất lợi ích. Đây là lý do thiết kế MAE đẩy mask token sang decoder để giảm mismatch.

