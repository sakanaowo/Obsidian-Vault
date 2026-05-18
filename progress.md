---
title: "D2L Learning Progress"
tags: [d2l, progress, deep-learning]
created: 2026-04-13
---

tác vụ: Dịch bài blog Colah "Understanding LSTM Networks" - 2026-04-19
nội dung: Đã tạo [[Understanding LSTM Networks - Colah]] trong 30_Resources/Webs/.
chi tiết:

- Dịch toàn bộ bài blog kinh điển của Christopher Olah (2015) sang tiếng Việt
- Giữ nguyên cấu trúc 7 phần gốc (RNN → Long-Term Dependencies → LSTM → Core Idea → Step-by-Step → Variants → Conclusion)
- Tải 16 hình minh họa gốc vào assets/attachments/colah-understanding-lstms/
- Thêm công thức toán (forget gate, input gate, cell update, output gate) cho mỗi bước
- Thêm ELI5, callout giải thích sâu (vanishing gradient, phép cộng cell state), bảng so sánh GRU vs LSTM
- Thêm ghi chú bối cảnh lịch sử 2015→2026 (Transformer, self-attention)
- Liên kết nội bộ tới [[LSTM]], [[GRU]], [[Backpropagation Through Time]], [[Buổi 42]], [[Buổi 43]]

---

tác vụ: D2L Learning - Mở rộng giải thích BPTT - 2026-04-19
nội dung: Đã cập nhật phần [[Buổi 42 - Tuần 12]], [[Buổi 43 - Tuần 12]] và ghi chú ngày 19-04 để giải thích BPTT kỹ hơn.
chi tiết:

- Thêm mục giải nghĩa bản chất của $x_t$, $h_t$, $o_t$, loss và gradient
- Viết lại phần ôn tập theo kiểu từ gốc đến nâng cao để dễ hiểu hơn
- Bổ sung active recall hôm nay với câu hỏi ngắn và đáp án tự kiểm tra
- Giải thích cụ thể vai trò của $W_{xh}$, $W_{hh}$, $W_{hq}$ và eigenvalue của $W_{hh}$
- Làm rõ `detach_()` chỉ cắt gradient, không xóa hidden state

---
tác vụ: D2L Learning - Tuần 12, Buổi 44 — 10.2 Gated Recurrent Units (GRU) - 2026-04-20
---
tác vụ: D2L Learning - Tuần 12, Buổi 45 — 10.3 & 10.4 Deep RNN + Bidirectional RNN - 2026-04-20
nội dung: Đã viết lại [[Buổi 45 - Tuần 12]] — Chapter 10.3 Deep RNN + 10.4 Bidirectional RNN (phiên bản giải thích sâu).
chi tiết:
- Deep RNN: stacking nhiều tầng RNN, mỗi tầng nhận output tầng dưới
- Công thức: H_t^(l) = phi(H_t^(l-1)W_xh + H_{t-1}^(l)W_hh + b) — 2 nguồn thông tin
- Inter-layer vs intra-layer: H_t^(l-1) (từ tầng dưới) vs H_{t-1}^(l) (từ bước trước)
- Siêu tham số: L in [1,8], h in [64,2056], ưu tiên tăng h trước L
- Cài đặt: StackedRNNScratch với nn.Sequential, forward layer-by-layer
- nn.GRU nhiều tầng: num_layers param, dropout giữa tầng, lr=2 thay vì lr=4
- Gradient clipping bắt buộc cho deep RNN, Layer Normalization thay vì BatchNorm
- Bidirectional RNN: 2 unidirectional chạy ngược chiều, output = concatenation [H→_t; H←_t]
- Forward H→_t đọc trái→phải (chứa x_1..x_t), Backward H←_t đọc phải→trái (chứa x_T..x_t)
- Shape: BiRNN output = 2h chiều, Deep BiRNN = 2^L × h chiều
- Hạn chế BiRNN: cần toàn bộ chuỗi → không real-time, chi phí gấp đôi
- Ứng dụng: POS tagging, NER, BERT pretraining, machine translation
- 2 hình D2L: deep-rnn-1.png (kiến trúc deep RNN), bi-rnn-1.png (kiến trúc BiRNN)
- 2 Mermaid diagrams bổ sung: deep RNN flow, BiRNN flow
- Active Recall: 10 câu ôn GRU (2 cổng, tổ hợp lồi, cell state, 25% params)
- Áp dụng quy tắc viết mới: tiếng Việt chủ đạo, giải thích kỹ khái niệm mới
---
nội dung: Đã tạo [[Buổi 44 - Tuần 12]] — Chapter 10.2 Gated Recurrent Units (GRU).
chi tiết:
- Bối cảnh lịch sử: Cho et al. (2014) đơn giản hóa LSTM thành GRU
- Kiến trúc 2 cổng: Reset Gate (R_t) + Update Gate (Z_t), bỏ cell state riêng
- Candidate hidden state: H̃_t = tanh(X_t W_xh + (R_t ⊙ H_{t-1}) W_hh)
- Công thức cốt lõi: H_t = Z_t ⊙ H_{t-1} + (1 - Z_t) ⊙ H̃_t — tổ hợp lồi
- So sánh LSTM vs GRU: 3 cổng vs 2, cell state riêng vs không, 4(dh+h²+h) vs 3(dh+h²+h)
- Cài đặt từ đầu: GRUScratch, triple() pattern, chỉ 1 hidden state H
- Cài đặt gọn: nn.GRU, so sánh 20 dòng vs 5 dòng
- Khi nào dùng GRU vs LSTM: chuỗi ngắn/tốc độ vs chuỗi dài/tinh vi
- 5 hình minh họa từ D2L: 2 cổng, candidate, hidden state, kiến trúc hoàn chỉnh, RNN/GRU/LSTM
- Active Recall: 10 câu ôn Buổi 43 LSTM (3 cổng, cell state, vanishing gradient, 4× params)
- Áp dụng quy tắc viết mới: tiếng Việt chủ đạo
---
tác vụ: D2L Learning - Tuần 12, Buổi 43 — 10.1 Long Short-Term Memory (LSTM) - 2026-04-20
nội dung: Đã tạo [[Buổi 43 - Tuần 12]] — Chapter 10.1 Long Short-Term Memory (LSTM). Bắt đầu Chapter 10 — Modern Recurrent Neural Networks.
chi tiết:

- Kiến trúc ô nhớ có cổng: 3 cổng (quên, đầu vào, đầu ra) + nút ứng viên + 2 trạng thái (C_t, H_t)
- Công thức cốt lõi: C*t = F_t ⊙ C*{t-1} + I_t ⊙ C̃_t — phép cộng tuyến tính giải quyết vanishing gradient
- So sánh gradient flow: RNN thường (W_hh^k → vanishing/exploding) vs LSTM (diag(F_t) → kiểm soát được)
- Cài đặt từ đầu: LSTMScratch với triple() pattern, forward nhận/trả bộ đôi (H, C)
- Cài đặt gọn: nn.LSTM wrapping, cuDNN fused kernel, so sánh 30 dòng vs 5 dòng
- Đếm tham số: 4(dh + h² + h) — gấp 4 lần RNN thường
- Exercises: hyperparameter tuning, GRU/LSTM/RNN chi phí so sánh, tanh lần 2 cho H_t, time series
- 5 hình minh họa matplotlib: cell architecture, gate behavior, cell state update, RNN vs LSTM gradient, formula summary
- Active Recall: 10 câu ôn Buổi 42 BPTT (gradient chain, truncation, eigenvalue, detach\_())
- Concept note mới: [[LSTM]] trong 20_Areas/AI/Concepts/ — giải quyết 15+ unresolved links
- Áp dụng quy tắc viết mới: tiếng Việt chủ đạo, không trộn lẫn ngôn ngữ gây lú

---

tác vụ: D2L Learning - Tuần 12, Buổi 42 — 9.7 Backpropagation Through Time - 2026-04-19
nội dung: Đã tạo [[Buổi 42 - Tuần 12]] — Chapter 9.7 Backpropagation Through Time (BPTT).
chi tiết:

- Section lý thuyết thuần (không có code mới) — nền tảng cho mọi hạn chế vanilla RNN
- Mô hình đơn giản hóa (9.7.1): h*t = f(x_t, h*{t-1}, w_h), gradient ∂L/∂w_h chứa tích Jacobian đệ quy
- Công thức then chốt (9.7.7): ∂h_t/∂w_h = Σ (Π ∂f/∂h) · ∂f/∂w_h — tích Jacobian gây vanishing/exploding
- 3 chiến lược: Full (O(T), impractical), Truncated (mặc định, biased nhưng ổn định), Randomized (unbiased, variance cao)
- BPTT chi tiết (9.7.2): identity activation, công thức (W_hh^T)^k → eigenvalue analysis
- Vanishing: |λ|<1 → gradient → 0. Exploding: |λ|>1 → gradient → ∞
- Kết nối code: detach\_() = truncated BPTT, giải thích lý do dùng ở Buổi 40-41
- Gradient clipping chỉ xử lý exploding, KHÔNG xử lý vanishing → motivate LSTM/GRU (Ch10)
- Exercises: orthogonal matrix eigenvalue proof, gradient alignment với eigenvector dominant, alternatives to clipping
- 5 hình minh họa matplotlib: computational graph, 3 truncation strategies, vanishing/exploding eigenvalue plot, gradient chain, detach visualization
- Active Recall: 10 câu ôn Buổi 41 (nn.RNN API, cuDNN, LazyLinear, swapaxes, 4 thay đổi, 2 bias, RNNCell)
- Concept note mới: [[Backpropagation Through Time]] trong 20_Areas/AI/Concepts/

tác vụ: D2L Learning - Tuần 11, Buổi 41 — 9.6 Concise Implementation of RNNs - 2026-04-16
nội dung: Đã tạo [[Buổi 41 - Tuần 11]] — Chapter 9.6 Concise Implementation of Recurrent Neural Networks.
chi tiết:

- So sánh 4 thay đổi chính: RNN core (nn.RNN), output layer (nn.LazyLinear), return format (stacked tensor), gradient clipping (auto)
- nn.RNN deep-dive: cuDNN fused kernel, tại sao nhanh hơn scratch (loại bỏ Python loop overhead)
- swapaxes(0,1) explained: time-first → batch-first convention mismatch
- nn.LazyLinear: tự suy luận input dim, giảm coupling giữa components
- nn.RNN parameters: weight_ih, weight_hh, bias_ih, bias_hh — so sánh 2 bias vs 1 bias scratch
- Data flow comparison: shape step-by-step (batch=1024, T=32, |V|=28, h=32)
- Training: cùng hyperparameters, comparable perplexity, faster execution
- Exercises: overfit strategies, autoregressive RNN implementation
- Khi nào dùng scratch vs high-level (+ nn.RNNCell trung gian)
- 4 hình minh họa matplotlib trong assets/attachments/d2l-buoi-41/
- Active Recall: 10 câu ôn Buổi 40 (RNNScratch, one-hot, gradient clipping, training, decoding)

tác vụ: D2L Learning - Tuần 11, Buổi 40 — 9.5 RNN Implementation from Scratch - 2026-04-15
nội dung: Đã tạo [[Buổi 40 - Tuần 11]] — Chapter 9.5 RNN Implementation from Scratch.
chi tiết:

- Implement RNNScratch: W_xh, W_hh, b_h với nn.Parameter, forward loop qua time steps
- One-hot encoding: tại sao cần (ordinal/distance bias), shape (batch,T)→(T,batch,|V|), transpose rationale
- Insight: one-hot × W_xh = chọn 1 hàng → W_xh là embedding matrix → nn.Embedding hiệu quả hơn
- RNNLMScratch: kết hợp RNN core + output layer (W_hq, b_q), full forward pass
- Gradient Clipping: ELI5, công thức g←min(1,θ/||g||)·g, Lipschitz continuity (Eq 9.5.1-9.5.3)
- So sánh gradient clipping vs giảm lr: adaptive intervention chỉ khi cần
- Training loop 7 bước: data → one-hot → RNN → output → loss → clip → update
- Hyperparameters: batch=1024, steps=32, hidden=32, lr=1, clip=1 — giải thích lr cao + clipping
- Decoding: warm-up phase (nạp prefix) + generation phase (argmax autoregressive)
- Greedy vs sampling decoding + temperature parameter
- Exercises: 10 bài giải chi tiết (num_steps, embedding, ReLU, bỏ clipping)
- 5 hình minh họa matplotlib trong assets/attachments/d2l-buoi-40/
- Active Recall: 8 câu ôn Buổi 39 + 10 câu Buổi 40
- Tạo concept note [[Gradient Clipping]] trong 20_Areas/AI/Concepts/

tác vụ: D2L Learning - Tuần 11, Buổi 39 — 9.4 Recurrent Neural Networks - 2026-04-14
nội dung: Đã tạo [[Buổi 39 - Tuần 11]] — Chapter 9.4 Recurrent Neural Networks.
chi tiết:

- Motivation: N-gram bùng nổ tham số O(|V|^n) → cần latent variable model
- So sánh MLP hidden layer vs RNN hidden state — phân biệt 2 khái niệm dễ nhầm
- Công thức cốt lõi RNN: H*t = tanh(X_t W_xh + H*{t-1} W_hh + b_h), O_t = H_t W_hq + b_q
- Unrolling theo thời gian + Weight Sharing: params cố định bất kể T
- Concatenation trick: [X_t, H_{t-1}] \* [W_xh; W_hh] — chứng minh + code demo
- Character-level LM: "machin" → "achine", softmax + cross-entropy mỗi time step
- Phân tích tham số: d=28, h=256, q=28 → ~80K params, W_hh chiếm 81.8%
- Tanh vs ReLU trong RNN: bounded, zero-centered, ổn định qua recurrence
- Vanishing/Exploding gradient: preview hướng giải quyết (LSTM, GRU, Transformer)
- 5 hình minh họa matplotlib trong assets/attachments/d2l-buoi-39/
- Active Recall: 8 câu ôn Buổi 38 + 10 câu chuyên sâu Buổi 39
- Tạo concept note [[Recurrent Neural Network]] trong 20_Areas/AI/Concepts/

tác vụ: D2L Learning - Tuần 10, Buổi 38 — 9.2 Converting Raw Text into Sequence Data & 9.3 Language Models - 2026-04-14
nội dung: Đã tạo [[Buổi 38 - Tuần 10]] — Chapter 9.2 + 9.3.
chi tiết:

- Pipeline tiền xử lý text 4 bước: Reading → Preprocessing → Tokenization → Vocab+Indexing
- 3 cấp độ tokenization: word-level, character-level, subword (BPE) — so sánh trade-offs
- Lớp Vocab: token_to_idx / idx_to_token, `<unk>` token, min_freq filtering
- Hàm build: pipeline hoàn chỉnh, character-level → corpus 173K tokens, vocab 28
- Zipf's Law: phân phối power law, log-log plot, ảnh hưởng đến N-gram sparsity
- Language Model: ước lượng P(x1,...,xT), chain rule decomposition
- N-gram Models: unigram/bigram/trigram, MLE counting, data sparsity problem
- Laplace Smoothing: thêm epsilon vào counts, uniform limit khi epsilon → ∞
- Perplexity: exp(cross-entropy trung bình), 3 cases (PP=1, PP=|V|, PP=∞)
- Sequence Partitioning: cắt corpus thành X/Y pairs (Y = X shifted by 1)
- 5 hình minh họa matplotlib trong assets/attachments/d2l-buoi-38/
- Active Recall: 7 câu ôn Buổi 37 + 10 câu chuyên sâu Buổi 38
- Tạo concept note [[Zipf's Law]] trong 20_Areas/AI/Concepts/

tác vụ: D2L Learning - Tuần 10, Buổi 37 — 9.1 Working with Sequences - 2026-04-13
nội dung: Đã tạo [[Buổi 37 - Tuần 10]] — Chapter 9.1 Working with Sequences.
chi tiết:

- Bước chuyển paradigm từ CNN (dữ liệu cố định) sang Sequence (dữ liệu tuần tự)
- 4 dạng bài toán Sequence: Seq→Fixed, Fixed→Seq, Aligned Seq→Seq, Unaligned Seq→Seq
- Autoregressive Models: Fixed window ($\tau$) vs Latent autoregressive (hidden state)
- Chain Rule decomposition — phân rã xác suất đồng thời thành tích xác suất có điều kiện
- Markov Models bậc 1, $k$, và mapping sang N-gram
- Thực nghiệm: Linear regression trên sin data, 1-step vs k-step prediction
- Error accumulation — sai số tăng hàm mũ, bài học interpolation vs extrapolation
- Stationarity assumption
- 5 hình ảnh minh họa trong assets/attachments/d2l-buoi-37/
- Active Recall: 8 câu ôn Buổi 36 (AnyNet/RegNet) + 10 câu chuyên sâu Buổi 37
- Tạo concept note [[Autoregressive Model]] trong 20_Areas/AI/Concepts/
- Fix link Buổi 36: "Buổi sau" từ [[Buổi 38]] → [[Buổi 37]] cho đúng số thứ tự

tác vụ: D2L Tổng ôn Buổi 25-36 — Builders Guide + CNN + Modern CNN - 2026-04-12
nội dung: Đã tạo [[Tổng ôn Buổi 25-36]] — Tổng ôn toàn diện từ Builders Guide đến Modern CNN.
chi tiết:

- Phạm vi: 12 buổi (25-36), kể từ lần tổng ôn gần nhất [[Tổng ôn Buổi 8-24]]
- Dạy lại (không chỉ recall) toàn bộ kiến thức: Save/Load, GPU, Conv, Pooling, Channels, LeNet, AlexNet, VGG, NiN, GoogLeNet, BN, ResNet, ResNeXt, DenseNet, RegNet
- 6 giai đoạn chính: Builders Guide → CNN Fundamentals → Modern CNN (9 kiến trúc) → Design Patterns
- Bảng tổng hợp 9 kiến trúc CNN + 5 design patterns xuyên suốt
- Công thức cốt lõi: Conv output size, params counting, BN, Residual gradient flow, DenseNet channels, Grouped conv
- 50 câu hỏi ôn tập + đáp án (7 nhóm: A-G) bao phủ mọi concept
- Checklist tự đánh giá 28 mục
- Mermaid diagrams: bản đồ kiến thức tổng thể + 5 design patterns

tác vụ: D2L Learning - Tuần 10, Buổi 36 - Designing CNN Architectures (AnyNet/RegNet) + Review - 2026-04-12
nội dung: Đã tạo [[Buổi 36 - Tuần 10]] — 8.8 Designing Convolution Network Architectures.
chi tiết:

- Viết toàn bộ nội dung theo cấu trúc ELI5-to-Deep (3 tầng) cho mỗi concept.
- Tạo và tích hợp 3 hình ảnh minh họa vào `assets/attachments/d2l-buoi-36/`:
  - `anynet_design_space.png` — AnyNet template: Stem → Body (4 stages) → Head, 4 tham số per-stage (§2.1)
  - `design_space_refinement.png` — Thu hẹp CDF: AnyNet_A → AnyNet_E qua 4 bước (§3.3)
  - `cnn_evolution_timeline.png` — Dòng tiến hóa CNN: LeNet → RegNet → ViT (§5.1)
- Active Recall đầu buổi (8 câu ôn kiến thức DenseNet từ Buổi 35).
- Active Recall chuyên sâu (10 câu + đáp án Claim/Reasoning/Evidence).
- AnyNet Design Space: template tổng quát, 17 free parameters, implementation PyTorch.
- Thu hẹp design space: CDF, 4 bước từ AnyNet_A → AnyNet_E, giảm từ 17 params tự do.
- RegNet: 4 nguyên tắc thiết kế (shared k, shared g, increasing c, increasing d), RegNetX-32.
- Discussion: CNN vs ViT, NAS vs Design Spaces.
- Tổng ôn Chapter 8: bảng so sánh 9 kiến trúc, 5 design patterns chung, Mermaid diagrams.
- Bảng thuật ngữ 11 mục + mapping D2L gốc.

tác vụ: Cập nhật concept node links - 2026-04-11
nội dung: Tạo 4 concept notes mới và cập nhật Concepts links trong Buổi 33, 34, 35.
chi tiết:

- Tạo [[Batch Normalization]] (20_Areas/AI/Concepts/) — ELI5, cơ chế toán, training vs inference, FC vs Conv
- Tạo [[Residual Connection]] (với alias [[Skip Connection]]) — degradation problem, gradient flow, addition vs concat
- Tạo [[Grouped Convolution]] — cơ chế nhóm, tiết kiệm params 1/G, cardinality trong ResNeXt
- Tạo [[Growth Rate (DenseNet)]] (với alias [[Growth Rate]]) — lý do k nhỏ hiệu quả, channel tracking formula
- Cập nhật Buổi 33 footer: thêm [[Batch Normalization]] vào Concepts
- Cập nhật Buổi 34 footer: thêm [[Residual Connection]], [[Skip Connection]], [[Grouped Convolution]] vào Concepts
- Cập nhật Buổi 35: xóa alias Growth Rate khỏi frontmatter (nhường cho concept note), thêm footer ## Liên kết đầy đủ

tác vụ: D2L Learning - Tạo mới Buổi 35 (DenseNet) - 2026-04-11
nội dung: Đã tạo [[Buổi 35 - Tuần 9]] — 8.7 Densely Connected Networks (DenseNet).
chi tiết:

- Viết toàn bộ nội dung theo cấu trúc ELI5-to-Deep (3 tầng) cho mỗi concept.
- Tạo và tích hợp 5 hình ảnh minh họa vào `assets/attachments/d2l-buoi-35/`:
  - `resnet_vs_densenet.png` — So sánh Addition vs Concatenation (§1.2)
  - `dense_block_flow.png` — Data flow và channel growth trong Dense Block (§2.4)
  - `transition_layer.png` — Cấu trúc Transition Layer (§4.2)
  - `densenet_architecture.png` — Kiến trúc tổng thể DenseNet (§5.1)
  - `resnet_densenet_comparison.png` — 3 khác biệt cốt lõi (§6.1)
- Active Recall đầu buổi (8 câu ôn kiến thức ResNet/ResNeXt từ Buổi 34).
- Active Recall chuyên sâu DenseNet (10 câu + đáp án Claim/Reasoning/Evidence).
- Mở rộng: DenseNet-BC (Bottleneck + Compression), bảng kiến trúc DenseNet-121/169/201/264.
- Bảng thuật ngữ (§10) và mapping với D2L gốc (§11).

tác vụ: D2L Learning - Viết lại toàn bộ Buổi 34 (ResNet & ResNeXt) - 2026-04-11
nội dung: Đã viết lại hoàn toàn [[Buổi 34 - Tuần 9]] với hình ảnh trực quan và Active Recall chuyên sâu cho ResNeXt.
chi tiết:

- Viết lại toàn bộ nội dung theo cấu trúc ELI5-to-Deep (3 tầng) cho mỗi concept.
- Tích hợp 7 hình ảnh minh họa từ `assets/attachments/d2l-buoi-34/` vào đúng ngữ cảnh:
  - `resnet_function_classes.png` — Non-nested vs Nested function classes (§1.2)
  - `resnet_residual_analogy.png` — Ẩn dụ họa sĩ: học f(x) vs học g(x) (§2.1)
  - `resnet_residual_block.png` — Regular vs Residual block (§2.1)
  - `gradient_flow_comparison.png` — Gradient flow có/không skip connection (§2.4)
  - `grouped_convolution.png` — Standard vs Grouped convolution (§5.2)
  - `resnext_bottleneck_flow.png` — Data flow qua ResNeXt bottleneck (§5.3)
  - `resnext_cost_comparison.png` — So sánh chi phí Standard vs ResNeXt (§5.4)
  - `resnet_vs_resnext_block.png` — So sánh cấu trúc 2 block (§5.5)
- Bổ sung Active Recall đầu buổi (8 câu ôn kiến thức cũ) và Active Recall chuyên sâu ResNeXt (10 câu + đáp án Claim/Reasoning/Evidence).
- Thêm bảng thuật ngữ đầy đủ (§9) và mapping với D2L gốc.

tác vụ: D2L Learning - thêm Active Recall vào Buổi 34 - 2026-04-09
nội dung: Đã cập nhật [[Buổi 34 - Tuần 9]] với mục `## Active Recall` theo rule mới.
chi tiết:

- Thêm 8 câu hỏi truy hồi cho kiến thức cũ (BatchNorm, NiN, GoogLeNet, Softmax/CE, Scratch vs Concise).
- Thêm phần tự trả lời ngắn gọn theo format "Claim -> Reasoning -> Evidence".
- Thêm danh sách concept notes cần ôn lại để chuẩn hóa review đầu buổi.

tác vụ: Cập nhật AGENTS - thêm rule Active Recall bắt buộc - 2026-04-09
nội dung: Đã cập nhật [[AGENTS]] để bắt buộc Active Recall ở mỗi buổi D2L.
chi tiết:

- Thêm rule mới: mọi buổi D2L phải có mục `## Active Recall` để ôn kiến thức cũ đã tạo.
- Quy định tối thiểu: 5-10 câu hỏi truy hồi, phần tự trả lời theo "Claim -> Reasoning -> Evidence", và danh sách concept links cần ôn.
- Bổ sung bước `Active Recall Check` vào workflow trước khi viết nội dung buổi mới.

tác vụ: D2L Learning - Bổ sung chuẩn hóa input vào Buổi 33 - 2026-04-07
nội dung: Đã cập nhật [[Buổi 33 - Tuần 9]] để nhắc lại chuẩn hóa input và sửa tham chiếu buổi học.
chi tiết:

- Gỡ tham chiếu cụ thể đến [[Buổi 24 - Tuần 7]] vì không khớp ngữ cảnh nguồn.
- Bổ sung callout "Ôn nhanh: chuẩn hóa input là gì?" với công thức $x'=(x-\mu_{train})/(\sigma_{train}+\epsilon)$.
- Làm rõ khác biệt: chuẩn hóa input áp dụng ở dữ liệu đầu vào, còn BatchNorm áp dụng cho activations bên trong mạng.

tác vụ: D2L Learning - Tuần 9, Buổi 33 - Batch Normalization - 2026-04-06
nội dung: Đã tạo [[Buổi 33 - Tuần 9]] — 8.5 Batch Normalization.
chi tiết:

- Tuân thủ rule Concept Introduction Structure 3 tầng (ELI5 → Định nghĩa kỹ thuật → Cơ chế).
- Tạo 2 ảnh minh họa: BN computation flow (4 bước + before/after distributions), BN cho FC vs Conv (normalize dimensions) → `assets/attachments/d2l-buoi-33/`.
- 3 lý do cần BN: preprocessing bên trong mạng, ổn định số học, regularization tự nhiên.
- Công thức BN đầy đủ: standardize → scale & shift, giải thích $\gamma, \beta$ learnable params.
- BN cho FC (dim=0) vs Conv (dim=0,2,3): bảng so sánh chi tiết, lý do translation invariance.
- So sánh BN vs Layer Normalization: bảng 6 tiêu chí, khi nào dùng cái nào.
- Training vs Prediction mode: 2 chế độ hoạt động, moving average, lỗi phổ biến quên model.eval().
- Implementation from scratch: hàm batch_norm + class BatchNorm, giải thích keepdim, momentum.
- BNLeNet: chèn BN vào LeNet, training với lr=0.1 (gấp 10x LeNet gốc).
- Concise: nn.LazyBatchNorm2d/1d, so sánh scratch vs concise.
- Discussion: Internal Covariate Shift (tranh cãi), landscape smoothing, xu hướng LN thay BN.
- 6 exercises gốc + 8 câu tự kiểm tra.
- Bảng thuật ngữ 12 mục.
- Mermaid diagrams dùng `<br>` thay `\n`, không emoji/icon.

tác vụ: D2L Learning - Tuần 8, Buổi 32 - GoogLeNet (Inception) - 2026-04-05
nội dung: Đã tạo [[Buổi 32 - Tuần 8]] — 8.4 Multi-Branch Networks (GoogLeNet).
chi tiết:

- Tuân thủ rule mới: Concept Introduction Structure 3 tầng (ELI5 → Định nghĩa kỹ thuật → Cơ chế).
- Tạo 2 ảnh minh họa: Inception block (4 nhánh), GoogLeNet architecture (Stem-Body-Head) → `assets/attachments/d2l-buoi-32/`.
- Inception block: 4 nhánh song song (1x1, 1x1+3x3, 1x1+5x5, MaxPool+1x1), concatenate channels.
- Phân tích bottleneck Conv 1x1: ví dụ cụ thể giảm params ~10x.
- Kiến trúc GoogLeNet: Stem (b1+b2) / Body (b3+b4+b5, 9 Inception blocks) / Head (GAP + 1 FC).
- Data flow analysis: 96→24→12→6→3→1, channels 64→192→480→832→1024.
- So sánh: GoogLeNet ~5M params vs VGG-16 ~138M (giảm ~28x).
- 5 exercises gốc + 8 câu tự kiểm tra.
- Bảng thuật ngữ 11 mục.
- Mermaid diagrams dùng `<br>` thay `\n`, không emoji/icon.

tác vụ: D2L Learning - Tuần 8, Buổi 31 - NiN (Network in Network) - 2026-04-05
nội dung: Đã tạo [[Buổi 31 - Tuần 8]] — 8.3 Network in Network (NiN).
chi tiết:

- Bám sát 100% cấu trúc d2l.ai 8.3, ELI5 → Deep ở mọi section.
- Tạo 3 ảnh minh họa: NiN block vs VGG block, Global Average Pooling, NiN architecture → `assets/attachments/d2l-buoi-31/`.
- 2 vấn đề của FC layers: chiếm ~90% params, không thể thêm vào giữa mạng.
- Conv 1×1 = FC per pixel: chứng minh toán học chi tiết, so sánh NiN block vs VGG block.
- Global Average Pooling: cơ chế hoạt động, tại sao cần num_classes channels, 0 params.
- Kiến trúc NiN đầy đủ: 4 NiN blocks + GAP, data flow analysis.
- So sánh NiN vs AlexNet vs VGG: params giảm ~60×.
- Di sản: conv 1×1 trong GoogLeNet/ResNet/MobileNet, GAP là default trong CNN hiện đại.
- 6 exercises gốc + 8 câu tự kiểm tra.
- Bảng thuật ngữ 12 mục.

tác vụ: D2L Learning - Tuần 8, Buổi 30 - VGG (Viết lại lần 2) - 2026-04-03
nội dung: Đã viết lại hoàn toàn [[Buổi 30 - Tuần 8]] — 8.2 Networks Using Blocks (VGG) với ảnh minh họa và nội dung mở rộng.
chi tiết:

- Tạo 3 ảnh minh họa: receptive field comparison, VGG-11 architecture, VGG block detail → `assets/attachments/d2l-buoi-30/`.
- Mở rộng phần VLSI analogy: thêm cột "Tương đồng", giải thích tại sao phép loại suy quan trọng.
- Mở rộng phần receptive field analysis: thêm công thức tổng quát $r = n(k-1) + 1$, giải thích chi tiết tại sao $k^2 c^2$ params.
- Mở rộng phần "Deep and Narrow > Shallow and Wide": chia rõ 3 lý do (ít params, nhiều ReLU → expressive power, implicit regularization), thêm giải thích ELI5 cho từng lý do.
- Cải thiện phần VGG family: giải thích tại sao VGG-11 và VGG-13 cùng ~133M params, phân tích tỉ lệ FC vs Conv params.
- Mở rộng phần Discussion: thêm trích dẫn gốc d2l.ai, phân biệt rõ "breakthrough" vs "blueprint".
- Nâng bài tự kiểm tra từ 7 → 8 câu, thêm câu về FC params percentage.
- Tuân thủ AGENTS.md: ELI5 → Deep ở mọi section, Mermaid + ảnh minh họa, bảng thuật ngữ 11 mục.

tác vụ: D2L Learning - Tuần 8, Buổi 30 - VGG (Networks Using Blocks) - 2026-04-02
nội dung: Đã tạo [[Buổi 30 - Tuần 8]] — 8.2 Networks Using Blocks (VGG).
chi tiết:

- Bám sát 100% cấu trúc d2l.ai 8.2, ELI5 → Deep.
- Bối cảnh: VLSI analogy, từ ad-hoc design → block-based design.
- Core insight: receptive field analysis — tại sao 2×Conv 3×3 > 1×Conv 5×5 (ít params, nhiều ReLU, regularize tốt hơn).
- VGG Block definition + VGG Network implementation (PyTorch).
- VGG family table (VGG-11/13/16/19) + so sánh chi tiết VGG vs AlexNet.
- Data flow analysis VGG-11: spatial halving + channel doubling pattern.
- Training trên Fashion-MNIST (phiên bản channels thu nhỏ).
- Discussion: VGG là CNN hiện đại đầu tiên "thực sự", hạn chế FC layers.
- 4 exercises gốc + 7 câu tự kiểm tra.

tác vụ: D2L Plan Restructure + Buổi 29 Rewrite - 2026-04-02
nội dung: Restructured D2L learning plan từ Buổi 29. Đã viết lại [[Buổi 29 - Tuần 8]] — 8.1 Deep Convolutional Neural Networks (AlexNet).
chi tiết:

- Xóa Buổi 29 cũ (nội dung quá tóm tắt, không bám sát d2l.ai).
- Cập nhật `D2L Learning Plan - 4 Months.md`: remapping Tuần 7-9 bám sát 1:1 theo d2l.ai chapters 7-8.
- Viết lại Buổi 29 hoàn toàn mới (~500 dòng), bám sát 100% cấu trúc d2l.ai 8.1:
  - Bối cảnh lịch sử: Feature Engineering → Representation Learning
  - Missing ingredients: Data (ImageNet), Hardware (GPU), Techniques (ReLU, Dropout, Xavier init)
  - Kiến trúc AlexNet chi tiết, so sánh AlexNet vs LeNet
  - Full implementation code PyTorch + training trên Fashion-MNIST
  - Discussion: Achilles heel (FC layers), overfitting paradox, adoption chậm
  - 8 bài tập gốc từ d2l + bài tự kiểm tra 7 câu
- Tuân thủ AGENTS.md: ELI5 → Deep, ví dụ cụ thể, Mermaid diagrams, bảng thuật ngữ.

tác vụ: D2L Learning - Tuần 5, Buổi 18 - Multilayer Perceptrons - 2026-03-24
nội dung: Đã viết [[Buổi 18 - Tuần 5]] (chapter_multilayer-perceptrons/mlp.md) — bước đầu vào Deep Learning.
chi tiết:

- Đã tạo note tại 10_Projects/D2L/Buổi 18 - Tuần 5.md theo chuẩn ELI5 → Deep.
- Đã tạo 2 concept notes mới: [[Activation Function]], [[Multilayer Perceptron]].
- Nội dung: giới hạn linear model, hidden layers, chứng minh affine collapse, 3 activation functions (ReLU, Sigmoid, Tanh), Universal Approximation Theorem.
- Đã thêm bảng từ điển 13 thuật ngữ + 5 câu tự kiểm tra.
- Image gen quota hết → dùng Mermaid diagrams thay thế.

tác vụ: D2L Learning - Tuần 4, Tổng ôn - 2026-03-23
nội dung: Đã tạo [[Tổng ôn Tuần 4]] — buổi ôn tập tổng hợp cho toàn bộ Tuần 4.
chi tiết:

- Bản đồ kiến thức Mermaid cho 5 buổi (13-17).
- Tóm tắt 1 dòng cho mỗi khái niệm theo từng buổi.
- 5 công thức cốt lõi cần nhớ (softmax, cross-entropy, gradient, SGD, LogSumExp).
- 6 sai lầm phổ biến kèm cách tránh.
- 4 bài trắc nghiệm + 3 bài tính tay + 3 bài code (tất cả có đáp án ẩn).
- Preview Tuần 5 (MLP).

tác vụ: D2L Learning - Tuần 4, Buổi 17 - Softmax Regression Concise + Review - 2026-03-23
nội dung: Đã viết [[Buổi 17 - Tuần 4]] (chapter_linear-classification/softmax-regression-concise.md) + Review Tuần 4.
chi tiết:

- Đã tạo note tại 10_Projects/D2L/Buổi 17 - Tuần 4.md.
- So sánh scratch vs concise, giải thích LogSumExp trick chi tiết.
- Đã thêm Mini Test 10 câu tổng ôn Tuần 4 (Buổi 13-17) kèm đáp án ẩn.
- Đã thêm Mermaid knowledge map cho Tuần 4.
- Đã thêm bảng từ điển 11 thuật ngữ.

tác vụ: D2L Learning - Tuần 4, Buổi 16 - Softmax Regression from Scratch - 2026-03-22
nội dung: Đã viết [[Buổi 16 - Tuần 4]] (chapter_linear-classification/softmax-regression-scratch.md) theo hướng nhập môn, kèm code đầy đủ.
chi tiết:

- Đã tạo note tại 10_Projects/D2L/Buổi 16 - Tuần 4.md theo chuẩn ELI5 → Deep cho người mới.
- Nội dung: implement softmax, cross-entropy, model class, training loop, error analysis.
- Đã thêm Mermaid flowchart cho training loop (image gen quota exhausted).
- Đã thêm bảng từ điển 13 thuật ngữ + 5 bài tập thực hành.
- Đã thêm code Python đầy đủ chạy được (Phần 6 — all-in-one).

tác vụ: D2L Learning - Tuần 4, Buổi 15 - Image Classification Dataset (Fashion-MNIST) - 2026-03-22
nội dung: Đã viết [[Buổi 15 - Tuần 4]] (chapter_linear-classification/image-classification-dataset.md) theo hướng nhập môn, kèm hình ảnh minh họa.
chi tiết:

- Đã tạo note tại 10_Projects/D2L/Buổi 15 - Tuần 4.md theo chuẩn ELI5 → Deep cho người mới.
- Đã tạo 2 hình ảnh: Fashion-MNIST samples overview, DataLoader pipeline diagram.
- Đã tạo 2 concept notes mới: [[Fashion-MNIST Dataset]], [[DataLoader (PyTorch)]].
- Đã thêm code PyTorch load + visualize Fashion-MNIST, giải thích tensor shape $(n,c,h,w)$.
- Đã thêm bảng từ điển 11 thuật ngữ + 5 câu tự kiểm tra.

tác vụ: Chuẩn hóa tiếng Việt có dấu cho Sigmoid note - 2026-03-19
nội dung: Đã chỉnh lại [[Sigmoid Function]] sang tiếng Việt có dấu đầy đủ.
chi tiết:

- Đã chuyển toàn bộ nội dung từ không dấu sang có dấu, giữ nguyên cấu trúc và công thức.
- Đã chuẩn hóa lại alias tiếng Việt: "hàm sigmoid".
- Không thay đổi ý nghĩa học thuật hay liên kết kiến thức liên quan.

tác vụ: Bổ sung concept Sigmoid Function - 2026-03-19
nội dung: Đã tạo [[Sigmoid Function]] trong thư mục Concepts.
chi tiết:

- Đã viết theo cấu trúc ELI5 -> First Principles -> Công thức -> Ứng dụng DL.
- Đã thêm các tính chất quan trọng: range, monotonicity, đối xứng, đạo hàm và saturation.
- Đã nối liên kết sang [[Softmax Function]], [[Cross-Entropy Loss]], [[Maximum Likelihood Estimation]].
- Đã bổ sung checklist tự kiểm tra và các lỗi thực hành thường gặp với BCE/BCEWithLogitsLoss.

tác vụ: Viết lại concept MLE theo First Principles - 2026-03-19
nội dung: Đã viết lại [[Maximum Likelihood Estimation]] theo cấu trúc ELI5 -> Deep.
chi tiết:

- Đã mở rộng từ định nghĩa ngắn sang giải thích bản chất likelihood và NLL.
- Đã thêm 2 cầu nối cốt lõi: Gaussian likelihood -> MSE, Categorical likelihood -> Cross-Entropy.
- Đã bổ sung phần bias-variance, consistency, asymptotic properties và liên hệ MLE vs MAP.
- Đã thêm checklist thực hành để chọn loss đúng theo phân phối dữ liệu.

tác vụ: D2L Learning - Tuần 4, Buổi 14 - Softmax Regression - 2026-03-19
nội dung: Đã viết [[Buổi 14 - Tuần 4]] (chapter_linear-classification/softmax-regression.md) theo hướng nhập môn, kèm hình ảnh minh họa.
chi tiết:

- Đã tạo note tại 10_Projects/D2L/Buổi 14 - Tuần 4.md theo chuẩn ELI5 → Deep cho người mới.
- Đã tạo 4 hình ảnh minh họa: softmax visualization, one-hot encoding, cross-entropy loss, softmax network diagram.
- Đã tạo 3 concept notes mới: [[Softmax Function]], [[Cross-Entropy Loss]], [[One-Hot Encoding]].
- Đã thêm bảng từ điển thuật ngữ 10 thuật ngữ + 5 câu tự kiểm tra.
- Nội dung bao gồm: Regression vs Classification, One-Hot Encoding, Linear Model cho multi-class, Softmax, Cross-Entropy từ MLE, Information Theory basics.

tác vụ: D2L Learning - Tuần 4, Buổi 13 - Generalization (Rewrite) - 2026-03-18
nội dung: Đã viết lại [[Buổi 13 - Tuần 4]] (chapter_linear-regression/generalization.md) theo hướng nhập môn, kèm hình ảnh minh họa.
chi tiết:

- Đã tạo note tại 10_Projects/D2L/Buổi 13 - Tuần 4.md theo chuẩn ELI5 → Deep cho người mới.
- Đã tạo 4 hình ảnh minh họa: polynomial curve fitting, complexity vs error U-curve, data split diagram, K-fold cross-validation.
- Đã tạo 3 concept notes mới: [[Training Error vs Generalization Error]], [[Overfitting and Underfitting]], [[Cross-Validation]].
- Đã cập nhật/mở rộng concept [[Generalization]] từ 30 dòng lên version đầy đủ với links.
- Đã thêm bảng từ điển thuật ngữ 12 thuật ngữ chuyên môn.
- Đã thêm 5 câu tự kiểm tra cuối buổi.

tác vụ: Viết lại Buổi 11 cho dễ hiểu hơn - 2026-03-17
nội dung: Đã viết lại [[Buổi 11 - Tuần 3]] theo hướng nhập môn rõ ràng, giảm độ nặng thuật ngữ.
chi tiết:

- Đã tái cấu trúc note theo flow đơn giản: 3 ý cốt lõi -> bài toán -> model -> loss -> SGD -> training loop.
- Đã thêm ví dụ số cụ thể ngay đầu buổi để người học thấy trực giác trước công thức.
- Đã giữ lớp ELI5 ở từng phần và tăng giải thích "vì sao làm bước này".
- Đã rút gọn phần tự kiểm tra để tập trung vào 4 câu quan trọng nhất.

---

tác vụ: D2L Learning - Tuần 3, Buổi 12 - Linear Regression Concise - 2026-03-17
nội dung: Đã hoàn thành [[Buổi 12 - Tuần 3]] (chapter_linear-regression/linear-regression-concise.md)
chi tiết:

- Đã tạo note mới tại 10_Projects/D2L/Buổi 12 - Tuần 3.md theo chuẩn ELI5 -> Deep, giải thích từ scratch sang concise.
- Đã map rõ 4 thành phần Scratch vs Concise: model/loss/optimizer/training loop.
- Đã giải thích chi tiết `nn.LazyLinear(1)`, `nn.MSELoss()`, `torch.optim.SGD` và vì sao bản chất toán không đổi.
- Đã thêm code PyTorch concise tối giản, checklist đọc kết quả và điểm dễ nhầm khi dùng API cấp cao.
- Đã thêm từ điển thuật ngữ buổi 12 để tránh nhầm giữa dịch nghĩa và nghĩa kỹ thuật.

---

tác vụ: D2L Learning - Tuần 3, Buổi 11 - Linear Regression from Scratch - 2026-03-16
nội dung: Đã hoàn thành [[Buổi 11 - Tuần 3]] (chapter_linear-regression/linear-regression-scratch.md + synthetic-regression-data.md)
chi tiết:

- Đã tạo note mới tại 10_Projects/D2L/Buổi 11 - Tuần 3.md theo format ELI5 -> Deep, giải thích cực chi tiết cho người mới.
- Đã phân rã rõ 4 khối cốt lõi: model, loss, optimizer (SGD), training loop.
- Đã thêm phần synthetic data với ground truth để kiểm thử implementation.
- Đã thêm code PyTorch tối giản, giải nghĩa từng bước và cách đọc output (loss, recover tham số).
- Đã bổ sung từ điển thuật ngữ riêng cho buổi 11 và danh sách lỗi thực tế cần tránh.

---

tác vụ: Viết lại Buổi 10 mức nhập môn sâu (giải thích từng phần) - 2026-03-16
nội dung: Đã viết lại [[Buổi 10 - Tuần 3]] theo hướng người mới hoàn toàn, tránh giả định đã biết thuật ngữ.
chi tiết:

- Đã thêm mục "Cách đọc file này" để người học có lộ trình đọc rõ ràng khi chưa nắm nền tảng.
- Đã mở rộng từng phần với ELI5 + giải nghĩa ký hiệu + ví dụ số cụ thể từng bước.
- Đã tăng độ chi tiết ở các phần MSE, SGD, Normal Equation, thêm so sánh MSE vs MAE bằng số.
- Đã bổ sung checklist tự đánh giá mức hiểu bài trước khi sang buổi tiếp theo.

---

tác vụ: Rewrite Buổi 10 (Dễ hiểu + Giải thích concept) - 2026-03-16
nội dung: Đã viết lại [[Buổi 10 - Tuần 3]] theo hướng dễ hiểu hơn, thêm mục giải thích các khái niệm xuất hiện trong buổi.
chi tiết:

- Đã tái cấu trúc nội dung theo flow: từ điển khái niệm -> mô hình -> loss -> tối ưu -> MSE-MLE -> code skeleton
- Đã thêm mục "Từ điển khái niệm" gồm 15 thuật ngữ cốt lõi (feature, label, bias, gradient, minibatch, MLE...)
- Đã đơn giản hóa diễn đạt công thức, giữ ký hiệu toán nhưng giảm độ trừu tượng trong phần giải thích
- Đã bổ sung ví dụ mini bằng số và danh sách điểm dễ nhầm để học nhanh
- Ghi chú: vẫn bám nguồn d2l linear-regression và linear-regression-scratch, nhưng ưu tiên khả năng đọc hiểu nhanh.

---

tác vụ: Rewrite Notes D2L - Buổi 8, 9, 10 - 2026-03-16
nội dung: Đã viết lại [[Buổi 8 - Tuần 2]], [[Buổi 9 - Tuần 2]], [[Buổi 10 - Tuần 3]] theo chuẩn sâu và bám sát d2l.ai.
chi tiết:

- Đã nâng cấp Buổi 8: làm rõ tư duy "programming with data", ranh giới rule-based vs ML, training loop chuẩn và mối nối với Preliminaries
- Đã nâng cấp Buổi 9: chuyển thành diagnostic review có đáp án kèm lập luận (Claim/Reasoning/Evidence), thêm tiêu chí qua checkpoint
- Đã nâng cấp Buổi 10: bổ sung các khối Deep Concept (Squared Loss, Normal Equation, MSE-MLE) theo tầng ELI5 -> bản chất -> công thức -> ứng dụng
- Đã giữ chuẩn frontmatter có `session:` cho cả 3 file
- Ghi chú: Timeline học đã liền mạch từ Buổi 7 -> 8 -> 9 -> 10, không còn nhảy buổi.

---

tác vụ: D2L Learning - Tuần 2, Buổi 9 - Review & Mini Test Preliminaries - 2026-03-15
nội dung: Đã hoàn thành [[Buổi 9 - Tuần 2]] (review week 1-2)
chi tiết:

- Đã review: Tensor ops, Linear algebra, Calculus, Autograd, Probability, Pandas, tư duy "programming with data"
- Đã tạo: Note review + mini test tại 10_Projects/D2L/Buổi 9 - Tuần 2.md
- Bài test: 8 câu tự kiểm tra (broadcasting, chain rule, Bayes, preprocessing, v.v.) + đáp án gợi ý
- Thời gian: ~1.5 giờ
- Ghi chú: Cần đặc biệt nhớ gradient accumulation trong PyTorch và vai trò base rate trong Bayes.

---

tác vụ: D2L Learning - Tuần 2, Buổi 8 - Introduction to Deep Learning - 2026-03-13
nội dung: Đã hoàn thành [[Buổi 8 - Tuần 2]] (chapter_introduction/index.md)
chi tiết:

- Đã học: Khi nào dùng rule-based vs machine learning, ví dụ wake-word, khái niệm model family/parameters/learning algorithm, training loop chuẩn
- Đã tạo: Note buổi học tại 10_Projects/D2L/Buổi 8 - Tuần 2.md
- Liên kết kiến thức: Tổng hợp mạch từ preliminaries sang linear models
- Thời gian: ~1.5 giờ
- Ghi chú: Tư duy cốt lõi là "programming with data"; đây là nền của mọi chương kỹ thuật tiếp theo.

---

tác vụ: D2L Learning - Tuần 3, Buổi 10 - Linear Regression - 2026-03-16
nội dung: Đã hoàn thành [[Buổi 10 - Tuần 3]] và tạo concept [[Linear Regression for Deep Learning]] (chapter_linear-regression/linear-regression.md + linear-regression-scratch.md)
chi tiết:

- Đã học: Mô hình affine $\hat{y}=w^Tx+b$, squared loss, nghiệm giải tích (Normal Equation), minibatch SGD, liên hệ MSE ↔ MLE dưới Gaussian noise
- Đã tạo: Note buổi học tại 10_Projects/D2L/Buổi 10 - Tuần 3.md
- Đã tạo: Concept Note tại 20_Areas/AI/Concepts/Linear Regression for Deep Learning.md
- Đã tạo bổ sung (tránh dead links): [[Maximum Likelihood Estimation]], [[Gradient Descent]], [[Generalization]]
- Bài tập tiếp theo: chứng minh bài toán tối ưu hằng số với MSE/MAE, chạy notebook linear-regression-scratch.md để kiểm tra recover tham số
- Thời gian: ~2 giờ
- Ghi chú: MSE có nền tảng xác suất rõ ràng khi giả định nhiễu cộng Gaussian; nghiệm đóng đẹp nhưng trong DL thực tế tối ưu lặp vẫn là phương pháp chính.

---

tác vụ: D2L Learning - Tuần 2, Buổi 7 - Data Preprocessing (Pandas) - 2026-03-11
nội dung: Đã hoàn thành [[Data Preprocessing with Pandas]] (chapter_preliminaries/pandas.md)
chi tiết:

- Đã học: pd.read_csv(), NaN handling, 3 loại Missing Data (MCAR/MAR/MNAR), Mean imputation (fillna), One-hot encoding (get_dummies + dummy_na), Deletion (dropna), Conversion to tensor (to_numpy → torch.tensor), iloc/loc indexing
- Đã tạo: Concept Note [[Data Preprocessing with Pandas]] trong 50_PTIT/DL/
- Bài tập: Load Abalone dataset, thực hành indexing bằng tên column
- Thời gian: ~1.5 giờ
- Ghi chú: MNAR là loại missing data nguy hiểm nhất — deletion gây selection bias. One-hot đúng về mặt toán học vì không giả định thứ tự giữa categories.

---

tác vụ: D2L Learning - Tuần 1, Buổi 6 - Probability & Statistics - 2026-03-09
nội dung: Đã hoàn thành [[Probability and Statistics for Deep Learning]] (chapter_preliminaries/probability.md)
chi tiết:

- Đã học: Sample space/events/Kolmogorov axioms, Random variables (discrete vs continuous), PMF/PDF, Joint/Marginal/Conditional probability, Bayes' theorem (HIV example), Independence & conditional independence, Expectation/Variance/Covariance matrix, Aleatoric vs Epistemic uncertainty, Chebyshev's inequality, liên hệ MLE ↔ Cross-entropy loss
- Đã tạo: Concept Note [[Probability and Statistics for Deep Learning]] trong 20_Areas/AI/Concepts/
- Bài tập: Tính variance của coin estimator, proof E[X-E[X]]=0, Markov chain factorization
- Thời gian: ~2 giờ
- Ghi chú: Base rate neglect (HIV test example) là bias rất phổ biến. L2 regularization = Gaussian prior — đây là cầu nối Bayesian ↔ frequentist DL. Law of Large Numbers hội tụ O(1/√n).

---

tác vụ: D2L Learning - Tuần 1, Buổi 5 - Automatic Differentiation - 2026-03-07
nội dung: Đã hoàn thành [[Automatic Differentiation]] (chapter_preliminaries/autograd.md)
chi tiết:

- Đã học: Computational graph (dynamic), PyTorch workflow (requires*grad/backward/.grad), gradient accumulation & zero*(), non-scalar backward (Jacobian/gradient arg), detach() & stop_gradient, control flow với if/while, dynamic vs static graph
- Đã tạo: Concept Note [[Automatic Differentiation]] trong 20_Areas/AI/Concepts/
- Bài tập: Plot sin(x) dùng autograd, trace dependency graph của f=(log x²·sin x)+x⁻¹
- Thời gian: ~2 giờ
- Ghi chú: PyTorch accumulates gradients (không auto-reset) — cần zero\_() trước mỗi backward. detach() quan trọng trong RL (target network) và self-supervised learning.

---

tác vụ: D2L Learning - Tuần 1, Buổi 4 - Calculus - 2026-03-07
nội dung: Đã hoàn thành [[Calculus for Deep Learning]] (chapter_preliminaries/calculus.md)
chi tiết:

- Đã học: Derivatives (định nghĩa, quy tắc), Partial derivatives, Gradient (vector partial derivatives), Matrix gradient identities (∇‖x‖²=2x, ∇Ax=Aᵀ), Chain rule (hàm 1 biến + nhiều biến), liên hệ với backprop
- Đã tạo: Concept Note [[Calculus for Deep Learning]] trong 20_Areas/AI/Concepts/
- Bài tập: Visualize f(x)=3x²-4x và tiếp tuyến tại x=1, làm exercises 1-5
- Thời gian: ~2 giờ
- Ghi chú: Chain rule là trái tim của backpropagation. ∇f = Jacobianᵀ · ∇_u y là dạng tổng quát. Saddle points là vấn đề thực tế trong training.

---

tác vụ: D2L Learning - Tuần 1, Buổi 3 - Linear Algebra - 2026-03-05
nội dung: Đã hoàn thành [[Linear Algebra for Deep Learning]] (chapter_preliminaries/linear-algebra.md)
chi tiết:

- Đã học: Scalars/Vectors/Matrices/Tensors hierarchy, Reduction (sum/mean/keepdims/cumsum), Hadamard product, Dot product, Matrix-vector product (mv/@), Matrix-matrix multiplication (mm/@), Norms (L1, L2, Frobenius)
- Đã tạo: Concept Note [[Linear Algebra for Deep Learning]] trong 20_Areas/AI/Concepts/
- Bài tập: Chạy code trong notebook linear-algebra.md, làm exercises
- Thời gian: ~2 giờ
- Ghi chú: Điểm then chốt — matrix multiply là nền tảng forward pass. Thứ tự nhân ma trận ảnh hưởng FLOPs. keepdims=True quan trọng cho broadcasting normalization.

---

tác vụ: D2L Learning - Tuần 1, Buổi 2 - Tensor Operations - 2026-03-04
nội dung: Đã hoàn thành [[Tensor Operations]] (chapter_preliminaries/ndarray.md)
chi tiết:

- Đã học: Tensor creation (arange, zeros, ones, randn), reshape, indexing & slicing, elementwise operations, broadcasting, memory efficiency, type conversion
- Đã tạo: Concept Note [[Tensor Operations]] trong 20_Areas/AI/Concepts/
- Bài tập: Chạy toàn bộ code trong notebook ndarray.md
- Thời gian: ~2 giờ
- Ghi chú: Broadcasting rules là phần cần nhớ kỹ (so sánh shape từ phải qua trái). In-place operations quan trọng cho training loop.

---

tác vụ: Tạo Notion Database cho Lịch Trình D2L - 2026-01-27
nội dung: Đã tạo Notion database "D2L Learning Schedule - 17 Weeks (Mar-Jul 2026)" với đầy đủ 17 tuần học.
chi tiết:

- **Database ID:** `90765f21-2799-43e9-8dc8-061ce80cf0b7`
- **URL:** https://www.notion.so/3cac3c9d41d342dc8c70c0c90cde3847
- **Schema Properties:**
  - Tuần (TITLE), Số Tuần (NUMBER), Giai Đoạn (SELECT with 9 phases)
  - Ngày Bắt Đầu/Kết Thúc (DATE), Số Buổi Học (NUMBER)
  - Mục Tiêu (RICH_TEXT), Trạng Thái (STATUS)
  - Test Milestone (CHECKBOX), Độ Khó (SELECT)
- **Pages Created:** 17 pages (Tuần 1 → Tuần 17) với content chi tiết
  - Tuần 1-2: Foundation (Beginner) - Test ở tuần 2
  - Tuần 3-6: Linear Models + MLPs (Beginner → Intermediate) - Tests ở tuần 4, 6
  - Tuần 7-9: CNNs (Intermediate → Advanced) - Test ở tuần 9
  - Tuần 10-11: RNNs (Intermediate → Advanced) - Test ở tuần 11
  - Tuần 12-13: Attention & Transformers (Advanced → Expert) ⭐ - Test ở tuần 13
  - Tuần 14-16: Advanced Topics (Expert)
  - Tuần 17: Review & Final Project - Final Test 🎉
- **Integration:** Sync với file Obsidian [[D2L Learning Plan - 4 Months]]
- **Next Step:** Sử dụng Notion để track progress hàng tuần (update Trạng Thái), Obsidian để viết notes chi tiết

---

tác vụ: Cập nhật Lịch Trình D2L - Tích hợp MIKU Agent - 2026-03-02
nội dung: Đã cập nhật [[D2L Learning Plan - 4 Months]] với hệ thống giảng dạy MIKU và workflow chi tiết.
chi tiết:

- **Thêm section MIKU Agent:** Giới thiệu phong cách giảng dạy ELI5
- **Workflow chi tiết:** 6 bước cho mỗi buổi học (Chuẩn bị → Lý thuyết với MIKU → Code → Bài tập → Kiểm tra → Progress update)
- **Milestone Tests:** Cập nhật 7 bài test lớn với format chuẩn, yêu cầu MIKU chi tiết
- **Quick Reference:** Thêm các câu lệnh thường dùng khi làm việc với MIKU
- **Tips học hiệu quả:** Cách đặt câu hỏi tốt, khi nào nên/không nên hỏi MIKU
- **Chiến lược:** Active recall, experiment-driven learning, first principles thinking
- **Tham chiếu:** Link đến `AGENTS.md` trong repo d2l-en

---

tác vụ: Tạo Lịch Trình Học D2L (4 Tháng) - 2026-03-02
nội dung: Đã tạo lộ trình học chi tiết cho cuốn sách [[Dive into Deep Learning]] trong 17 tuần (02/03/2026 → 01/07/2026).
chi tiết:

- **File:** [[D2L Learning Plan - 4 Months]]
- **Cấu trúc:** 17 tuần, 62 buổi học, phủ toàn bộ 23 chapters của d2l-en
- **Milestone Tests:** 6 bài test lớn + 1 final project
- **Phân bổ:**
  - Tuần 1-2: Preliminaries (ndarray, calculus, probability, autograd)
  - Tuần 3-4: Linear Models (Regression + Classification)
  - Tuần 5-6: MLPs + Builders Guide
  - Tuần 7-9: CNNs (LeNet → ResNet/DenseNet)
  - Tuần 10-11: RNNs (Vanilla → LSTM/GRU → Seq2Seq)
  - Tuần 12-13: Attention & Transformers (CORE!)
  - Tuần 14-15: Optimization + NLP Applications
  - Tuần 16: Specialization (chọn 1: GANs/RL/GPs/RecSys)
  - Tuần 17: Review + Final Project
- **Integration:** Kết hợp với repo d2l-en tại `30_Resources/Books/Dive Into Deep Learning/d2l-en/`
- **Tracking:** Mỗi buổi học sẽ cập nhật vào `progress.md` và tạo concept notes trong `20_Areas/AI/Concepts/`

---

tác vụ: Tiền xử lý và tạo note Chương 1 - Giáo trình Lịch sử Đảng CSVN - 2026-01-26
nội dung: Đã tách PDF (221 trang) thành 4 chương và viết note chi tiết cho Chương 1.
chi tiết:

- **Tách PDF**: Chương nhập môn (12 trang), Chương 1 (47 trang), Chương 2 (56 trang), Chương 3 (106 trang)
- **Note Chương 1**: [[Chương 1 - Đảng CSVN Ra Đời và Lãnh Đạo Đấu Tranh Giành Chính Quyền]]
  - I. Đảng CSVN ra đời và Cương lĩnh chính trị đầu tiên (2-1930)
  - II. Lãnh đạo đấu tranh giành chính quyền 1930-1945 (Xô viết Nghệ-Tĩnh, phong trào dân chủ, Cách mạng Tháng Tám)
- Output PDF: `assets/attachments/giao-trinh-lich-su-dang/`
- Script mới: `plugins/pdf tools/split_lich_su_dang.py`

---

tác vụ: Penn Treebank exercises (DailyNote) - 2026-01-20
nội dung: Đã bổ sung đáp án Penn Treebank POS + parse tree cho [[20-01-2026]] và mở rộng cơ chế trong [[Part-of-Speech Tagging]].
chi tiết:

- Gán POS tag cho từng token, xác định subject/main verb, liệt kê noun phrases.
- Vẽ Penn Treebank phrase-structure parse cho “Students study NLP.” và trả lời câu hỏi phân tích (relative clause/head noun/VP elements).
- Vẽ Penn Treebank phrase-structure parse cho câu có SBAR (relative clause) “Students who study NLP …” và phân tích NP/VP/PP/SBAR.
- Bổ sung cột NOTE giải thích vai trò ngữ pháp cho từng token trong bài 2.2.
- Đổi diagram parse tree sang `mermaid` để nhìn rõ trong Obsidian.
- Bổ sung bảng giải thích ≥ 8 POS tags (kèm tag không xuất hiện trong câu để đủ yêu cầu).
- Thêm mục “POS tagging hoạt động như thế nào? (ELI5 → Deep)” để làm rõ pipeline và mô hình (HMM/CRF/Transformer).

---

tác vụ: Dịch NLP Book PTIT (Chapter 2-4) - 2026-01-19
nội dung: Đã dịch và viết lại 3 chapters của tài liệu NLP PTIT theo chuẩn Deep & Comprehensive.
chi tiết:

- **Chapter 2 - POS Tagging:** [[Chapter 2]] — HMM, Viterbi Algorithm, Beam Search
- **Chapter 3 - Language Models:** [[Chapter 3]] — N-gram, Perplexity, Smoothing techniques
- **Chapter 4 - Sentiment Classification:** [[Chapter 4]] — Naive Bayes, Evaluation metrics
- **Concept Notes tạo mới:**
  - [[Hidden Markov Model]], [[Viterbi Algorithm]], [[Markov Chain]]
  - [[N-gram Language Model]], [[Perplexity]], [[Smoothing (NLP)]]
  - [[Naive Bayes]]
- Tích hợp ảnh từ `assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/`

---

---

tác vụ: MCP Concept Notes (AI) - 2026-01-18
nội dung: Đã tạo cụm concept notes về [[Model Context Protocol (MCP)]] và các thành phần liên quan, kèm MOC điều hướng.
chi tiết:

- Tạo mới: [[Model Context Protocol (MCP)]], [[MCP Server]], [[MCP Client]], [[MCP Resources]], [[MCP Tools]], [[MCP Transports]], [[MCP Security Model]].
- Tạo mới để tránh dead link và làm rõ bề mặt tấn công: [[Prompt Injection]].
- Tạo MOC: [[Agent Tooling & Integration MOC]] để liên kết MCP với [[Function Calling]] và [[LangChain]].

---

tác vụ: Clean up duplicate files - 2025-12-14
nội dung: Đã dọn dẹp các file trùng lặp trong thư mục `30_Resources`.
chi tiết:

- Di chuyển `Vector Databases.md` sang `20_Areas/AI/Concepts`.
- Tạo `20_Areas/Data/Concepts` và di chuyển `JSON.md`, `YAML.md` vào đó.
- Xóa các file concept trùng lặp trong các thư mục con của `30_Resources/Books/Prompt_Engineering_Phoenix_Taylor/` (01_Foundations, 02_Models_Architecture, 03_Techniques_Strategies, 05_Challenges).
- Xóa các thư mục rỗng đã dọn dẹp.

---

---

tác vụ: Rewrite Chapter 03 of Prompt Engineering Book - 2025-12-14
nội dung: Đã viết lại [[Chapter 03 - Standard Practices for Text Generation with ChatGPT]].
chi tiết:

- Tích hợp nội dung chi tiết từ PDF về các kỹ thuật tạo văn bản (List, JSON, YAML, Chunking, CoT, v.v.).
- Thêm các hình ảnh minh họa (Fig 3-1 đến Fig 3-9) từ thư mục assets.
- Cấu trúc lại theo định dạng chuẩn Zettelkasten/MOC.

---

---

tác vụ: Protocol Adoption - 2025-12-14
trạng thái: Active
nội dung: Chính thức áp dụng Bộ Quy Tắc Mới (Protocol v2.0) cho Agent.
quy_tắc_mới:

1. NO-BULLET-POINT: Giải thích concept bằng đoạn văn chuyên sâu (Deep Paragraphs), hạn chế gạch đầu dòng.
2. ATOMIC KNOWLEDGE: Tách định nghĩa sâu sang `20_Areas`, Book Note chỉ chứa context và link.
3. ASSET CHECK-FIRST: Luôn kiểm tra file ảnh tồn tại trong `assets/` trước khi trích xuất lại.
4. SHOW, DON'T JUST TELL: Bắt buộc trích dẫn code/diagram và phân tích sâu.

---

---

tác vụ: Refactor Chapter 03 (Future)
trạng thái: completed
nội dung: Viết lại Chapter 03 theo tiêu chuẩn Protocol v2.0.
các_bước:

- Audit: Đã hoàn tất kiểm tra các khái niệm cốt lõi.
- Extract: Đã hoàn tất tách nội dung kỹ thuật và tạo [[Hallucination]], [[Divide Labor]], [[Sentiment Analysis]] concept notes.
- Rewrite: Đã hoàn tất viết lại Book Note Chapter 03 theo Protocol v2.0.
- Cleanup: Đã hoàn tất kiểm tra và xác nhận không có file trùng lặp hoặc rác phát sinh từ quy trình này.

---

---

tác vụ: Process Chapter 04 & Workflow Update - 2026-01-04
trạng thái: completed
nội dung: Cập nhật workflow xử lý PDF và hoàn thành Chapter 04 (Deep Rewrite).
chi tiết:

- **New Rule:** Đã cập nhật `GEMINI.md` với quy trình sử dụng `pdf_chapter_extractor.py` cho tài liệu dài.
- **Extraction:** Đã tách toàn bộ PDF "Prompt Engineering..." thành 10 chapters riêng biệt.
- **Concepts:** Đã tạo mới các concept quan trọng: [[LangChain]], [[Function Calling]], [[Prompt Chaining]], [[Few-Shot Prompting]], [[LLM Evaluation]].
- **Book Note:** Đã hoàn thành [[Chapter 04 - Advanced Techniques for Text Generation with LangChain]] với nội dung chuyên sâu, phân tích kỹ thuật (LCEL, Map-Reduce, Agents) và tích hợp code example.

---

---

tác vụ: Process Chapter 05 - 2026-01-04
trạng thái: completed
nội dung: Hoàn thành xử lý Chapter 05 về Vector Databases.
chi tiết:

- **Extraction:** Đã trích xuất hình ảnh cho Chapter 5.
- **Concepts:** Đã tạo/cập nhật sâu các concept: [[Embeddings]], [[Semantic Search]], [[Vector Databases]], [[FAISS]], [[Pinecone]].
- **Book Note:** Đã hoàn thành [[Chapter 05 - Vector Databases with FAISS and Pinecone]] theo chuẩn Deep & Comprehensive, so sánh chi tiết giữa FAISS và Pinecone, giải thích cơ chế Indexing.

---

---

tác vụ: Process Chapter 06 - 2026-01-04
trạng thái: completed
nội dung: Hoàn thành xử lý Chapter 06 về Autonomous Agents.
chi tiết:

- **Extraction:** Đã trích xuất hình ảnh (ReAct framework, Memory, BabyAGI architecture).
- **Concepts:** Đã tạo mới các concept nền tảng: [[Autonomous Agents]], [[ReAct Framework]], [[Plan-and-Execute Agents]], [[Agent Memory]], [[Agent Toolkits]], [[BabyAGI]], [[AutoGPT]].
- **Book Note:** Đã hoàn thành [[Chapter 06 - Autonomous Agents with Memory and Tools]] với phân tích sâu về cơ chế hoạt động của Agent và vai trò của Memory/Tools.

---

---

tác vụ: Process Chapter 07 - 2026-01-04
trạng thái: completed
nội dung: Hoàn thành xử lý Chapter 07 về Image Generation Models (Diffusion).
chi tiết:

- **Rule Update:** Đã cập nhật `GEMINI.md` với quy tắc "Extreme Detail" và "References".
- **Extraction:** Đã trích xuất hình ảnh quan trọng.
- **Concepts:** Đã xây dựng hệ thống concept note chuyên sâu: [[Diffusion Models]], [[Latent Space]], [[CLIP]], [[VAE]], [[U-Net]].
- **Papers:** Đã tạo thư mục `30_Resources/Papers` và thêm các Source Note cho các bài báo nền tảng (Stable Diffusion, CLIP, DDPM, DALL-E 2, Imagen).
- **Book Note:** Đã hoàn thành [[Chapter 07 - Introduction to Diffusion Models for Image Generation]] với phân tích kiến trúc và liên kết đầy đủ tới tài liệu tham khảo.

---

---

tác vụ: Rewrite Chapter 01 of NLP Book - 2026-01-06
trạng thái: completed
nội dung: Viết lại [[Chapter 01 - Foundations of Machine Learning]] theo chuẩn Protocol mới.
chi tiết:

- **Math Foundations:** Cập nhật kiến thức về Linear Algebra, Probability, và Information Theory (Entropy, KL Divergence).
- **Modeling:** Phân biệt rõ ràng Generative (Naive Bayes) và Discriminative (Logistic Regression) models.
- **Challenges:** Thêm phân tích về OOV, Smoothing, Inductive Bias và Non-linearity.
- **Links:** Đã liên kết tới các concept notes quan trọng ([[Entropy (Information Theory)]], [[KL Divergence]]).

---

---

tác vụ: Complete Rewrite of Chapter 01 (NLP Book) - 2026-01-06
trạng thái: completed
nội dung: Hoàn tất viết lại toàn bộ [[Chapter 01 - Foundations of Machine Learning]].
chi tiết:

- **Part 2 Added:** Tích hợp nội dung từ trang 31-60.
- **Advanced Loss:** Thêm Ranking loss, Contrastive loss, Error-based loss.
- **Model Selection:** Thêm Bias-Variance Tradeoff, Regularization, Cross-validation.
- **Task Mapping:** Bảng tổng hợp ánh xạ các bài toán NLP (Classification, Seq Labeling, Seq2Seq) sang mô hình ML.
- **New Links:** Đã liên kết tới [[Bias-Variance Tradeoff]], [[LLM Evaluation]].

---

---

tác vụ: Deep Rewrite of Chapter 01 (Correction) - 2026-01-06
trạng thái: completed
nội dung: Viết lại toàn bộ [[Chapter 01 - Foundations of Machine Learning]] theo yêu cầu "Deep & Comprehensive".
chi tiết:

- **Correction:** Khắc phục vấn đề nội dung quá ngắn và thiếu chi tiết trong lần thử trước.
- **Expansion:** Mở rộng giải thích Toán học (Vectors, Probability, Entropy) thành các đoạn văn sâu.
- **Deep Dive:** Phân tích chi tiết cơ chế Naive Bayes vs Logistic Regression, OOV, Smoothing, Inductive Bias.
- **Completeness:** Bao phủ toàn bộ 60 trang, bao gồm Loss Functions, Regularization, Evaluation, và Task Mapping.
- **Format:** Chuyển từ bullet points sang deep paragraphs theo đúng quy tắc.

---

---

tác vụ: Restart Chapter 01 (Part 1 - Math Primitives) - 2026-01-06
trạng thái: completed
nội dung: Viết lại [[Chapter 01 - Foundations of Machine Learning]] (10 trang đầu) sang Tiếng Việt.
chi tiết:

- **Language Fix:** Đã chuyển toàn bộ nội dung sang Tiếng Việt chuyên ngành theo đúng quy tắc dự án.
- **Scope:** Tập trung sâu vào 10 trang đầu (Math Primitives).
- **Content:** Giải thích chi tiết về Đại số tuyến tính (Vectơ, Ma trận, Chuẩn) và Xác suất (Phân phối, Kỳ vọng, Phương sai).

---

---

tác vụ: Continue Chapter 01 (Part 2 - Text Classification) - 2026-01-06
trạng thái: completed
nội dung: Cập nhật và Viết tiếp Phần 2 cho [[Chapter 01 - Foundations of Machine Learning]] (Trang 11-20).
chi tiết:

- **Rollback:** Đã khôi phục nội dung Phần 1 (Math Primitives).
- **Bridge:** Bổ sung/làm rõ phần Kỳ vọng (Expectation) để nối mạch lạc.
- **New Content:** Thêm chi tiết về Lý thuyết thông tin (Entropy), Bài toán Phân loại Văn bản (BoW, Linear Classifiers) và so sánh Generative vs Discriminative.

---

---

tác vụ: Process PTIT NLP PDF (Chapter 1) - 2026-01-13
nội dung: Đã tách PDF và tích hợp kiến thức vào [[Chapter 1]].
chi tiết:

- Tách `assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf` thành 8 chapter PDF trong `assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang_chapters/`.
- Trích xuất ảnh theo từng chapter vào `assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_<n>/`.
- Viết lại `50_PTIT/NLP/Chapter 1.md` theo hướng dịch + phân tích (First Principles), có đánh dấu “Suy luận thêm”.
- Tạo các concept notes để tránh dead links và chuẩn hóa tri thức: [[Natural Language Processing (NLP)]], [[NLP Pipeline]], [[Sentence Segmentation]], [[Stemming]], [[Lemmatization]], [[Stop Words]], [[Dependency Parsing]], [[Part-of-Speech Tagging]], [[Ambiguity (NLP)]], [[NLP Challenges]].

---

---

tác vụ: Dịch và trình bày lại paper 2305 & 2306 - 2026-01-13
nội dung: Đã tạo Source Notes cho [[LIMA - Less Is More for Alignment]] và [[Textbooks Are All You Need]].
chi tiết:

- Viết lại nội dung paper theo hướng dịch + phân tích (First Principles), nhấn mạnh cơ chế và hàm ý thực nghiệm.
- Tạo concept notes để tránh dead links và chuẩn hóa thuật ngữ: [[Instruction Tuning]], [[Supervised Fine-Tuning (SFT)]], [[Reinforcement Learning from Human Feedback (RLHF)]], [[Synthetic Data (LLM Training)]], [[Data Curation]], [[Superficial Alignment Hypothesis]].

---

---

tác vụ: Dịch paper MAE (He et al., CVPR 2022) - 2026-01-14
nội dung: Đã tạo/viết lại [[Masked Autoencoders Are Scalable Vision Learners]].
chi tiết:

- Đã trích xuất ảnh từ PDF vào `assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/` để minh họa hiện tượng tái tạo dưới masking ratio cao.
- Đã lưu bản **nguyên văn** (pdftotext) trong [[Masked Autoencoders Are Scalable Vision Learners (Original)]] và viết bản dịch + phân tích theo First Principles.
- Đã tạo các concept notes để tránh dead links và chuẩn hóa thuật ngữ: [[Masked Autoencoders (MAE)]], [[Vision Transformers (ViT)]], [[Self-Supervised Learning (Computer Vision)]], [[Autoencoders]], [[Image Patches]], [[Masking Ratio]], [[Linear Probing]], [[Fine-Tuning (Transfer Learning)]].

---

---

tác vụ: Fix format bản nguyên văn MAE - 2026-01-14
nội dung: Đã sửa lỗi format trong [[Masked Autoencoders Are Scalable Vision Learners (Original)]].
chi tiết:

- Loại bỏ ký tự ngắt trang (form feed `\\x0c`) do `pdftotext` sinh ra, tránh lỗi render trong Obsidian.

---

---

tác vụ: Reformat bản nguyên văn MAE (Obsidian-friendly) - 2026-01-14
nội dung: Đã chỉnh lại [[Masked Autoencoders Are Scalable Vision Learners (Original)]] để đọc ổn trong Obsidian.
chi tiết:

- Chuyển bản trích xuất từ `pdftotext -layout` sang `pdftotext` (không layout) để tránh lỗi hiển thị cột/dẫn tới “grid” dày đặc trong code block.

---

---

tác vụ: Cập nhật quy tắc ELI5 trong AGENT - 2026-01-14
nội dung: Đã cập nhật [[AGENT]] để bắt buộc “Explain Like I'm 5” cho mọi phần giải thích.
chi tiết:

- Thêm chuẩn 2 tầng: `ELI5 → Deep` để vừa dễ hiểu ngay vừa đảm bảo phân tích học thuật.

---

---

tác vụ: Viết lại MAE Paper Note (Comprehensive Rewrite) - 2026-01-14
nội dung: Đã viết lại hoàn toàn [[Masked Autoencoders Are Scalable Vision Learners]] theo chuẩn "First Principles" với từng đoạn đối chứng văn bản gốc.
chi tiết:

- Cấu trúc mới: Mỗi phần bao gồm (1) Quote gốc tiếng Anh, (2) Dịch nghĩa Tiếng Việt, (3) Giải thích sâu/phân tích.
- Đã thêm các bảng kết quả ablation (Masking Ratio, Decoder Design, Mask Token, Reconstruction Target, Data Augmentation, Mask Sampling Strategy).
- Đã tích hợp hình ảnh minh họa từ `assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/` (Figure 1 architecture, reconstruction examples, mask sampling strategies).
- Đã thêm công thức toán học để giải thích cơ chế (attention complexity, MSE loss, masking ratio effect).
- Đã liên kết tới các Concept Notes: [[Masked Autoencoders (MAE)]], [[Vision Transformers (ViT)]], [[Self-Supervised Learning (Computer Vision)]], [[BERT]], [[Contrastive Learning]].

---

---

tác vụ: Tạo Concept Notes cho MAE Paper - 2026-01-14
nội dung: Đã tạo 10 concept notes mới để giải thích các thuật ngữ/khái niệm quan trọng trong bài MAE.
chi tiết:

- [[BERT]]: Masked Language Model, so sánh với MAE (masking ratio 15% vs 75%)
- [[GPT]]: Autoregressive Language Model, scaling laws
- [[Contrastive Learning]]: SimCLR, MoCo, so sánh với MAE (discriminative vs generative)
- [[Denoising Autoencoders]]: Ancestor của MAE, cơ chế corruption
- [[BEiT]]: BERT for Vision, so sánh dVAE tokens vs pixels
- [[Positional Embedding]]: Sinusoidal vs Learned, vai trò trong MAE decoder
- [[Mean Squared Error]]: Loss function, normalized vs unnormalized pixels
- [[Transfer Learning]]: MAE vs supervised pre-training performance
- [[ImageNet]]: Dataset và benchmark, MAE state-of-the-art kết quả
- [[Data Augmentation]]: Tại sao MAE không cần augmentation mạnh

---

---

tác vụ: Dịch toàn bộ MAE Paper với ELI5 - 2026-01-16
nội dung: Đã viết lại hoàn toàn [[Masked Autoencoders Are Scalable Vision Learners]] theo yêu cầu dịch từng section với giải thích ELI5.
chi tiết:

- Cấu trúc 11 phần chính: Abstract, Introduction, Related Work, Approach, ImageNet Experiments, Comparisons, Transfer Learning, Discussion, Reconstruction Examples, Contributions, Concept Links.
- Mỗi phần có dịch thuật kèm ELI5 trong `> [!TIP]` callout để giải thích cho người chưa có background.
- Đã tích hợp tất cả bảng ablation: Masking Ratio, Decoder Design, Mask Token, Reconstruction Target, Data Augmentation, Mask Sampling Strategy.
- Đã embed các hình minh họa từ `assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/` (Figure 1, 2, 3, 4, 5, 6).
- Đã liên kết tới 16 Concept Notes để tránh dead links và chuẩn hóa thuật ngữ.
- Format theo mẫu LIMA và Textbooks papers trong vault.

---

tác vụ: D2L Learning - Tuần 13, Buổi 48 — 10.7 Sequence-to-Sequence Learning - 2026-04-22
nội dung: Đã tạo [[Buổi 48 - Tuần 13]] — 10.7 Sequence-to-Sequence Learning (phiên bản giải thích sâu).
chi tiết:

- 10 phần toàn diện bám sát d2l.ai 10.7: Overview, Teacher Forcing, Encoder, Decoder, Seq2Seq Model, Masked Loss, Training, Prediction, BLEU, Bottleneck/Attention
- Hai paper nền tảng: Cho et al. (2014) + Sutskever et al. (2014), tại sao reverse source sequence
- Teacher forcing chi tiết: ELI5, so sánh vs free-running, exposure bias, 4 giải pháp
- Seq2SeqEncoder: Xavier init, embedding + GRU, shape analysis, forward pass
- Seq2SeqDecoder: concat context, input_size = embed+hidden, dense projection
- Data flow minh họa: từng bước context cố định → bottleneck
- Masked loss: masking strategy, ví dụ số, so sánh ignore_index vs manual mask
- Training loop: gradient clipping bắt buộc, shift-right, Adam optimizer
- Prediction: greedy decoding, error accumulation, motivation beam search
- BLEU: precision n-gram, brevity penalty, ví dụ tính tay, implementation
- Bottleneck analysis: context vector cố định → attention là giải pháp
- 6 exercises gốc từ D2L + 6 bài tự kiểm tra
- 6 hình minh họa matplotlib trong assets/attachments/d2l-buoi-48/ (seq2seq layers, teacher forcing, greedy decoding, BLEU score, masked loss, architecture overview)
- Active Recall: 10 câu ôn Buổi 47 (Encoder-Decoder, shift-right, masked loss, teacher forcing, BLEU)
- Áp dụng quy tắc viết mới: tiếng Việt chủ đạo, ELI5 ở mỗi concept, 3 tầng structure
- Cập nhật AGENTS.md: Python env luôn dùng conda `d2l`

---

tác vụ: D2L Learning - Tuần 13, Buổi 49 — 10.8 Beam Search - 2026-04-23
nội dung: Đã tạo [[Buổi 49 - Tuần 13]] — 10.8 Beam Search (phiên bản giải thích sâu).
chi tiết:

- Fix ảnh d2l-buoi-46: 4 file .svg thực chất là HTML → tạo lại bằng matplotlib
  - seq2seq.svg, encoder.svg, decoder.svg, mt-seq2seq.svg, mt-transformer.svg
  - Xóa page.html (236KB HTML, không phải ảnh)
  - Chỉ giữ lại rnn.svg (đúng định dạng)
- 3 hình minh họa beam search (d2l-buoi-49/):
  - greedy-search.svg: Greedy vs Non-Greedy path với ví dụ số cụ thể
  - exhaustive-search.svg: Cây tìm kiếm $|\mathcal{Y}|^{T'}$ branches
  - beam-search-comparison.svg: So sánh 3 chiến lược Greedy vs Beam vs Exhaustive
- Tải ảnh D2L: beam-search.svg (beam k=2, max_len=3, định dạng SVG hợp lệ)
- 6 phần toàn diện bám sát d2l.ai 10.8:
  - Phần I: Bài toán giải mã sequence, ký hiệu
  - Phần II: Greedy Search (ELI5, ví dụ số 0.048 vs 0.054, ưu/nhược)
  - Phần III: Exhaustive Search (cây, chi phí $V^{T'}$, tại sao không khả thi)
  - Phần IV: Beam Search (ELI5 k bạn đồng hành, thuật toán, ví dụ k=2)
  - Phần V: Code PyTorch Beam Search class + shape analysis
  - Phần VI: Chọn beam size, length normalization chi tiết, sampling so sánh
  - Phần VII: Bảng tóm tắt, Phần VIII: Exercises
  - Phần IX: Active Recall + liên kết Buổi 46-47-48-49
- Giải thích sâu: tại sao Greedy thất bại (local vs global optimum)
- Log-probability: biến tích thành tổng, tránh underflow
- Length normalization: $\frac{1}{L^\alpha}\sum \log P$, tại sao $\alpha=0.75$
- Exercises: beam size calculations, log probability computation, beam search vs GPT sampling
- Active Recall: 5 câu hỏi ôn tập từ Buổi 46-47-48
- Fix Buổi 48 link: "Tuần 14" → "Tuần 13" (2 occurrences)

---

tác vụ: Viết lại [[Buổi 49 - Tuần 13]] + Tạo [[Tổng ôn RNN]] - 2026-04-23
nội dung: Đã viết lại [[Buổi 49 - Tuần 13]] theo chuẩn mới (deep-learning-notes skill) và tạo [[Tổng ôn RNN]] tổng hợp toàn bộ Chương 9 + 10.
chi tiết:

- Viết lại Buổi 49 (Beam Search) hoàn chỉnh theo chuẩn skill: ELI5, định nghĩa kỹ thuật (WHAT/WHY/Input-Output), từ điển ký hiệu, so sánh, code với comment từng dòng, Reader Checklist
- Tổng ôn RNN: 12 buổi (38→49), 3 phần chính: Vanilla RNN (N-gram → RNN → BPTT), Modern RNN (LSTM → GRU → Deep/BiRNN), Seq2Seq (Encoder-Decoder → Beam Search)
- ELI5 và tự trả lời cho mỗi concept
- Anti-cramming checks cho: weight sharing, gradient clipping, log probability, ignore_index, cell state update (LSTM), convex combination (GRU)
- 50+ câu hỏi ôn tập (Nhóm A-F), đáp án mẫu cho 3 câu đầu mỗi nhóm
- Bảng tổng hợp toàn bộ RNN family, Reader Checklist, Tự đánh giá checklist
- Tạo skill mới: .cursor/skills/deep-learning-notes/ (SKILL.md + PEDAGOGY.md + NOTE_CONVENTIONS.md)

---

tác vụ: D2L Learning - Tuần 14, Buổi 50 — 11.1 Queries, Keys, and Values (viết lại) - 2026-04-23
nội dung: Viết lại [[Buổi 50 - Tuần 14]] — 11.1 Queries, Keys, and Values (rewrite hoàn chỉnh, đối chiếu với D2L source).
chi tiết:

- Đối chiếu note cũ với D2L.ai 11.1 source, phát hiện và bổ sung 5 gaps: (1) Motivation fixed-size input problem, (2) Summary/Nadaraya-Watson connection, (3) 4 D2L exercises, (4) D2L Fig 11.1.1, (5) convex cone vs convex combination distinction
- Thêm Part I: Motivation — từ RNN bottleneck (Buổi 48-49) sang attention, ví dụ số 10 tokens × 256 hidden → mất 90% thông tin
- Bổ sung connection quan trọng: Attention = Differentiable Nadaraya-Watson estimator (Nadaraya 1964, Watson 1964) — cầu nối sang Buổi 51
- Tạo 3 hình mới (gen_buoi50_figures_v2.py): d2l-fig-11-1-1 (D2L-style attention mechanism flow), attention-special-cases (4-panel heatmap), attention-database-analogy (Traditional DB vs Attention DB)
- Phân biệt rõ: Convex cone (chỉ nonnegative) vs Convex combination (nonnegative + sum=1) — tránh nhầm lẫn khi học
- 4 bài tập D2L 11.1.3: hard attention, chứng minh gradient = covariance, differentiable search engine, SE Networks
- Giữ ELI5 ở mỗi section, từ điển ký hiệu đầy đủ, Reader Checklist, Active Recall (ôn Buổi 49 + 6 câu mới)

---

tác vụ: D2L Learning - Tuần 14, Buổi 51 — 11.2 Attention Pooling by Similarity (Nadaraya-Watson) - 2026-04-25
nội dung: Đã tạo [[Buổi 51 - Tuần 14]] — 11.2 Attention Pooling by Similarity.
chi tiết:

- Giải thích Nadaraya-Watson estimator như tiền thân non-parametric của attention mechanism (Nadaraya 1964, Watson 1964)
- 4 kernel functions (D2L Eq. 11.2.1): Gaussian, Boxcar, Epanechikov, Constant — với hình dạng và đặc điểm
- NW Regression Formula (D2L Eq. 11.2.2): $f(q) = \sum_i v_i \cdot \frac{\alpha(q, k_i)}{\sum_j \alpha(q, k_j)}$
- Dataset demo: $y_i = 2\sin(x_i) + x_i + \epsilon$ với 40 training points
- Implementation nadaraya_watson() function: tính ma trận khoảng cách, kernel scores, normalize, weighted sum
- Quan sát quan trọng: Gaussian, Boxcar, Epanechikov cho kết quả gần như giống nhau — kernel shape ít quan trọng bằng việc có kernel
- Gaussian width $\sigma$ effect: narrow $\sigma$ → local adaptation (overfit noise), wide $\sigma$ → smooth (underfit)
- Bias-variance trade-off: classic balance giữa fitting local patterns và stability
- Tạo 6 hình minh họa: kernel-shapes (4 kernels), nw-regression-comparison (4 kernels vs ground truth), nw-attention-weights (4 heatmaps), nw-gaussian-width (4 sigma values), nw-query-diagram (attention lines), nw-vs-learned-attention (hand-crafted vs learned comparison)
- Active Recall: 5 câu ôn Buổi 50 (QKV, softmax, convex combination, differentiability) + 5 câu mới về NW
- 6 bài tập D2L 11.2.5: Parzen windows equivalence, SGD for kernel width, MSE minimization, leave-one-out, unit sphere simplification → dot-product attention, consistency và tốc độ giảm scale
- Connection: Unit sphere simplification $\|\mathbf{x}-\mathbf{x}_i\|^2 = 2 - 2\mathbf{x}^\top\mathbf{x}_i$ → Gaussian kernel reduces to scaled dot-product → foundation của dot-product attention (Buổi 52)

---

tác vụ: D2L Learning - Tuần 14, Buổi 52 — 11.3 Attention Scoring Functions - 2026-04-25
nội dung: Đã tạo [[Buổi 52 - Tuần 14]] — 11.3 Attention Scoring Functions.
chi tiết:

- Derivation: Gaussian kernel → Dot Product (D2L 11.3.1): khai triển $-\frac{1}{2}\|q-k\|^2 = q^T k - \frac{1}{2}\|k\|^2 - \frac{1}{2}\|q\|^2$, 3 terms biến mất sau normalization và layer norm
- Scaled Dot Product: tại sao cần chia $\sqrt{d}$ (D2L 11.3.2) — variance analysis: Var$(q^T k / \sqrt{d}) = 1$ khi q, k i.i.d. $\sim \mathcal{N}(0,1)$. Không chia → Var $= d$ → softmax saturation → gradient vanish
- Masked Softmax (D2L 11.3.2.1): xử lý variable-length sequences bằng valid_lens, cơ chế -1e6, 1D vs 2D valid_lens formats
- BMM — Batch Matrix Multiplication (D2L 11.3.2.2): shape analysis, parallel computation
- DotProductAttention class: Q @ K^T / √d → masked_softmax → dropout → A @ V. Shapes: (batch,n,d) × (batch,m,d) → (batch,n,m) → (batch,n,v)
- Additive Attention (D2L 11.3.4): cho q ≠ k dimensions, $a(q,k) = w_v^T \tanh(W_q q + W_k k)$, Bahdanau-style scoring
- So sánh Dot Product vs Additive: 0 params vs MLP params, q=k vs q≠k, O(n·m·d) vs O(n·m·h)
- Tạo 6 hình: d2l-fig-11-3-1, scaled-dot-product, masked-softmax, dot-product-vs-additive, batch-matrix-multiplication, dot-product-forward-shapes
- Active Recall: 5 câu ôn Buổi 51 (NW, kernel, variance, unit sphere) + 6 câu mới về scoring functions
- 3 bài tập D2L 11.3.6: distance-based attention, different dimensions, complexity analysis

---

tác vụ: Viết lại Buổi 51 và Buổi 52 theo chuẩn template D2L - 2026-04-27
nội dung: Đã viết lại [[Buổi 51 - Tuần 14]] và [[Buổi 52 - Tuần 14]] theo đúng chuẩn template D2L.
chi tiết:

- Thống nhất "Trả lời nhanh" thay vì "Tự trả lời" (5 câu truy hồi + 5 câu trả lời)
- Active Recall format: **Claim → Reasoning → Evidence** (3 phần)
- Mỗi Active Recall: đổi "### Tự trả lời" thành "### Trả lời nhanh"
- Chuẩn hóa thuật ngữ: "attention weights" → "trọng số chú ý (attention weights)"
- Chuẩn hóa thuật ngữ: "shape" → "kích thước (shape)" trong các bảng và text
- Chuẩn hóa thuật ngữ: "dropout" giữ nguyên (đây là thuật ngữ phổ biến)
- Thêm bảng thuật ngữ đầy đủ ở cuối mỗi buổi với định nghĩa tiếng Việt
- Đảm bảo tất cả thuật ngữ xuất hiện lần đầu đều có dịch tiếng Việt
- Thêm "Thay thế/gợi ý giải pháp nào trước đây?" trong định nghĩa kỹ thuật
- Thêm Reader Checklist sau mỗi section quan trọng (buổi 51: 5 checklists, buổi 52: 4 checklists)
- Thêm từ điển ký hiệu đầy đủ cho mỗi công thức toán
- Chuẩn hóa heading structure: PHẦN I, II, III... → PHẦN I, II, III... (không có khoảng trắng thừa)

---

tác vụ: D2L Learning - Tuần 14, Buổi 54 — 11.5 Multi-Head Attention - 2026-05-02
nội dung: Đã tạo [[Buổi 54 - Tuần 14]] — 11.5 Multi-Head Attention.
chi tiết:

- Active Recall: 5 câu ôn Buổi 53 (Bahdanau QKV, bottleneck, additive scoring, context vector, encoder-decoder vs self-attention)
- Giải thích tại sao cần nhiều heads — single-head chỉ học được 1 loại dependency
- Ví dụ: "The bank of the river" — 1 head không thể capture cả syntax và semantics
- Công thức cho một head (D2L Eq. 11.5.1): $\mathbf{h}_i = f(\mathbf{W}_i^{(q)}\mathbf{q}, \mathbf{W}_i^{(k)}\mathbf{k}, \mathbf{W}_i^{(v)}\mathbf{v})$
- Công thức output cuối cùng (D2L Eq. 11.5.2): $\mathbf{o} = \mathbf{W}_o [\mathbf{h}_1; ...; \mathbf{h}_h]$
- Tại sao chia chiều cho h: giữ tổng chiều = d, giảm computation từ $d^2$ xuống $d^2/h$
- Implementation chi tiết: MultiHeadAttention class với 4 linear layers (W_q, W_k, W_v, W_o)
- transpose_qkv: reshape để tính attention song song cho nhiều heads
- transpose_output: reverse của transpose_qkv
- valid_lens xử lý: repeat_interleave cho mỗi head
- So sánh: Single-head vs Multi-head vs Bahdanau attention
- Sơ đồ minh họa multi-head attention flow
- Bảng thuật ngữ đầy đủ: multi-head attention, head, subspace, concatenation

