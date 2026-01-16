---
type: concept
title: GPT
aliases:
  - Generative Pre-trained Transformer
  - Autoregressive Language Model
tags:
  - ai
  - nlp
  - transformers
  - language-models
---

**GPT (Generative Pre-trained Transformer)** là một họ mô hình ngôn ngữ **tự hồi quy** (autoregressive) do OpenAI phát triển, bắt đầu từ GPT-1 (Radford et al., 2018). Ý tưởng cốt lõi: pre-train một Transformer decoder bằng cách dự đoán **token tiếp theo** dựa trên các token trước đó, sau đó fine-tune cho downstream tasks.

**Cơ chế Autoregressive**

Khác với [[BERT]] (nhìn cả hai chiều), GPT sử dụng **causal masking**: mỗi token chỉ "nhìn" được các token trước nó trong chuỗi. Mục tiêu huấn luyện là tối đa hóa likelihood:
$$
\mathcal{L} = \sum_{t} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1})
$$

Điều này cho phép GPT **sinh văn bản** một cách tự nhiên: bắt đầu từ một prompt, dự đoán token tiếp theo, thêm vào chuỗi, lặp lại.

**Sự tiến hóa của GPT**

| Version | Params | Đặc điểm |
|---------|--------|---------|
| GPT-1 | 117M | Chứng minh pre-training + fine-tuning works |
| GPT-2 | 1.5B | Zero-shot capabilities, "too dangerous to release" |
| GPT-3 | 175B | In-context learning, few-shot prompting |
| GPT-4 | ~1T+ | Multimodal, reasoning, emergent abilities |

**So sánh với BERT và MAE**

GPT và BERT đều là phương pháp pre-training tự giám sát trong NLP, nhưng có triết lý khác nhau:

| Aspect | GPT | BERT |
|--------|-----|------|
| Kiến trúc | Decoder-only | Encoder-only |
| Hướng attention | Causal (trái→phải) | Bidirectional |
| Mục tiêu | Next token prediction | Masked token prediction |
| Ứng dụng chính | Generation | Understanding (classification, NER) |

Trong vision, [[Masked Autoencoders (MAE)]] lấy cảm hứng từ cả hai:
- Từ BERT: ý tưởng **masked prediction** (che và dự đoán)
- Nhưng giải quyết các thách thức riêng của vision: masking ratio cao hơn, pixel reconstruction thay vì token

**iGPT: GPT cho hình ảnh**

Trước MAE, **iGPT** (Chen et al., 2020) đã thử áp dụng autoregressive modeling cho ảnh: coi ảnh như chuỗi pixel và dự đoán pixel tiếp theo. Tuy nhiên, iGPT gặp vấn đề compute (phải xử lý chuỗi rất dài) và chất lượng representation không bằng contrastive methods thời đó. MAE chọn hướng masked prediction (như BERT) thay vì autoregressive (như GPT), và giải quyết vấn đề compute bằng asymmetric encoder-decoder.

**Scaling Laws**

GPT chứng minh **scaling laws**: performance tăng theo power law khi tăng model size, data size, và compute. Điều này thúc đẩy cuộc chạy đua xây dựng mô hình ngày càng lớn. MAE cũng thể hiện scaling behavior tương tự trong vision: ViT-H (86.9%) > ViT-L (85.9%) > ViT-B (83.6%).
