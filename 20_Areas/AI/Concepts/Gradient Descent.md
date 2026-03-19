---
title: "Gradient Descent"
aliases: [GD, tối ưu gradient]
tags: [concept, optimization, deep-learning]
created: 2026-03-16
---

# Gradient Descent

> [!NOTE] ELI5
> Bạn đang đứng trên đồi và muốn xuống điểm thấp nhất. Gradient cho biết hướng dốc lên nhanh nhất, nên bạn đi ngược lại hướng đó để đi xuống. Mỗi bước đi nhỏ gọi là một lần cập nhật tham số.

Cập nhật chuẩn:

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t)
$$

- $\theta$: tham số mô hình.
- $L(\theta)$: hàm mất mát.
- $\eta$: learning rate.

Biến thể thường gặp:

- Batch GD: dùng toàn bộ dữ liệu mỗi bước.
- SGD: dùng 1 mẫu mỗi bước.
- Minibatch SGD: dùng một lô nhỏ, cân bằng tốc độ và ổn định.

## TODO

- [ ] Bổ sung hình học trực quan về learning rate quá lớn/quá nhỏ.
- [ ] Thêm liên hệ với momentum, RMSProp, Adam.
- [ ] Bổ sung điều kiện hội tụ cho hàm lồi và không lồi.
