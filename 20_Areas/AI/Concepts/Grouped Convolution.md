---
title: "Grouped Convolution"
aliases: ["Group Convolution", "Grouped Conv", "tích chập nhóm"]
tags: [concept, deep-learning, architecture, efficiency, resnext]
created: 2026-04-11
---

# Grouped Convolution (Tích chập Nhóm)

> [!NOTE] ELI5
> Tưởng tượng một lớp học 60 học sinh cần được kèm cặp bởi 1 giáo viên — giáo viên đó phải chú ý đến tất cả mọi người cùng lúc, rất mệt. **Grouped convolution** giống như chia lớp thành 3 nhóm 20 người, mỗi nhóm có 1 giáo viên riêng làm việc song song — nhanh hơn và hiệu quả hơn. Mỗi giáo viên (conv trong nhóm) chỉ cần "chú ý" tới 1/3 số channels, giảm đáng kể công việc.

**Grouped Convolution** là biến thể của convolution trong đó $C_{in}$ input channels được chia thành $G$ nhóm (groups) bằng nhau ($C_{in}/G$ channels/nhóm). Mỗi nhóm thực hiện convolution độc lập với $C_{out}/G$ output filters. Kết quả của $G$ nhóm được concatenate lại. Kết quả này làm giảm số tham số và FLOPs theo hệ số $1/G$ so với standard convolution.

## Cơ chế

**Standard convolution:** Mỗi filter nhìn toàn bộ $C_{in}$ channels:
$$\text{Params} = C_{in} \times C_{out} \times k_h \times k_w$$

**Grouped convolution** ($G$ groups): Mỗi filter chỉ nhìn $C_{in}/G$ channels:
$$\text{Params}_G = \frac{C_{in}}{G} \times \frac{C_{out}}{G} \times k_h \times k_w \times G = \frac{C_{in} \times C_{out} \times k_h \times k_w}{G}$$

**Tiết kiệm:** Grouped conv có $1/G$ số params và FLOPs so với standard conv.

## Ví dụ số

Conv layer: $C_{in} = 128$, $C_{out} = 128$, kernel $3 \times 3$, $G = 32$:

- Standard: $128 \times 128 \times 9 = 147{,}456$ params
- Grouped ($G=32$): $\frac{128}{32} \times \frac{128}{32} \times 9 \times 32 = 4{,}608$ params
- **Giảm 32 lần!**

## Trường hợp đặc biệt

| $G$              | Tên gọi            | Ý nghĩa                                       |
| ---------------- | ------------------ | --------------------------------------------- |
| $G = 1$          | Standard conv      | Tất cả channels kết nối với nhau              |
| $G = C_{in}$     | **Depthwise conv** | Mỗi channel có 1 filter riêng (MobileNet)     |
| $1 < G < C_{in}$ | Grouped conv       | Trade-off giữa hiệu quả và khả năng biểu diễn |

## Ứng dụng trong ResNeXt

**Cardinality** trong ResNeXt = số groups $G$ trong grouped convolution của bottleneck block. Tăng cardinality (nhiều nhóm hơn) hiệu quả hơn tăng depth hoặc width ở cùng chi phí tính toán — đây là phát hiện chính của bài báo ResNeXt (Xie et al., 2017).

## Liên kết

- Đã học chi tiết ở [[Buổi 34 - Tuần 9]] (ResNeXt — cardinality)
- Liên quan: [[Residual Connection]], [[Batch Normalization]]
- Source: [d2l.ai — 8.6 ResNet and ResNeXt](https://d2l.ai/chapter_convolutional-modern/resnet.html)

---

> [!TODO]
>
> - Depthwise Separable Convolution (MobileNet): depthwise + pointwise
> - Channel Shuffle (ShuffleNet): khắc phục vấn đề groups không giao tiếp với nhau
> - Tại sao cardinality hiệu quả hơn depth/width? Phân tích từ góc độ optimization landscape
