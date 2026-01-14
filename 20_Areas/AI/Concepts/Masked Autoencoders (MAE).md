---
type: concept
title: Masked Autoencoders (MAE)
aliases:
  - MAE (Masked Autoencoders)
tags:
  - ai
  - computer-vision
  - self-supervised-learning
  - transformers
---

**Masked Autoencoders (MAE)** là một phương pháp **tự giám sát** cho thị giác máy tính thuộc họ [[Autoencoders]]/**denoising autoencoders**, trong đó ta **che (mask)** một phần lớn tín hiệu đầu vào rồi yêu cầu mô hình tái tạo lại phần bị che. Biến thể MAE nổi bật (He et al., CVPR 2022) kết hợp hai ý tưởng mang tính “cơ học” hơn là “mẹo”: (i) **masking ratio cao** (ví dụ 75% patch) để biến bài toán tái tạo thành nhiệm vụ đòi hỏi suy luận toàn cục; (ii) kiến trúc **encoder–decoder bất đối xứng**, trong đó encoder chỉ chạy trên patch nhìn thấy, còn decoder nhẹ xử lý đầy đủ token để tái tạo.

Từ góc nhìn xác suất, MAE đang học một bài toán gần với “mô hình hóa có điều kiện”: dự đoán $X_{mask}$ từ $X_{vis}$, tức xấp xỉ $p(X_{mask}\\mid X_{vis})$. Khi masking ratio thấp, bài toán dễ rơi vào shortcut “nội suy texture” vì ảnh có tính dư thừa cao; representation khi đó dễ thiên về thống kê thấp tầng. Khi masking ratio cao và mask phân bố ngẫu nhiên, phần bị che trở thành thiếu hụt thông tin đáng kể, buộc mô hình phải học cấu trúc đối tượng/cảnh để tái tạo hợp lý. Đây là cơ chế khiến MAE có thể học **ngữ nghĩa** dù mục tiêu vẫn là pixel.

Kiến trúc bất đối xứng giải quyết một nút thắt compute: encoder backbone (thường là [[Vision Transformers (ViT)]]) có self-attention bậc hai theo số token. Nếu ảnh có $N$ patch và che tỉ lệ $r$, encoder chỉ xử lý $(1-r)N$ token nên chi phí attention giảm xấp xỉ theo $(1-r)^2$. Với $r=0.75$, riêng phần attention lý thuyết giảm ~16×. Thực nghiệm paper cho speedup ~3–4× vì decoder vẫn chạy full token và còn nhiều chi phí khác, nhưng đây là “đòn bẩy” để scale encoder lớn mà vẫn huấn luyện thực tế được.

Một chi tiết thiết kế thường bị hiểu nhầm: MAE **không** đưa mask token vào encoder. Điều này giảm “pretrain–deploy mismatch”: downstream ta feed ảnh đầy đủ patch, nên tốt hơn nếu encoder đã quen với patch thật thay vì một phần lớn token giả. Mask token được đưa vào **sau** encoder và chỉ tồn tại trong decoder nhằm phục vụ tái tạo.

Trong thực hành, MAE pretrain xong sẽ **vứt decoder** và dùng encoder cho downstream bằng [[Fine-Tuning (Transfer Learning)]], hoặc đánh giá representation bằng [[Linear Probing]]. Vì vậy, khi đọc ablation của MAE, cần tách rõ “decoder tối ưu cho reconstruction” và “encoder tối ưu cho recognition”: decoder đủ mạnh có thể hấp thụ tính chuyên biệt của tái tạo, giúp encoder giữ latent trừu tượng hơn.

