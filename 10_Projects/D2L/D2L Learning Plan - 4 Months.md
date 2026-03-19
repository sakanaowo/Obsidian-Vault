---
title: "Lịch Trình Học D2L - 4 Tháng (02/03/2026 → 01/07/2026)"
tags: [learning-plan, deep-learning, d2l, study-schedule]
created: 2026-03-02
duration: 17 weeks
goal: "Hoàn thành toàn bộ Dive into Deep Learning với kiến thức vững chắc từ cơ bản đến nâng cao"
---

# 📚 Lịch Trình Học D2L - 4 Tháng

> [!NOTE] ELI5 - Tổng quan
> Đây là lộ trình học **Dive into Deep Learning** được thiết kế như một khóa học đại học 4 tháng. Mỗi tuần bạn sẽ học từ 3-5 buổi, mỗi buổi khoảng 2-3 giờ. Cuối mỗi chương lớn sẽ có bài kiểm tra để củng cố kiến thức.

## 🎯 Mục tiêu chung

- ✅ **Nắm vững lý thuyết:** Hiểu sâu các khái niệm từ Linear Regression đến Transformers
- ✅ **Thực hành code:** Chạy và hiểu toàn bộ notebooks trong repo d2l-en
- ✅ **Áp dụng thực tế:** Hoàn thành ít nhất 3 Kaggle competitions trong sách
- ✅ **Tư duy hệ thống:** Biết khi nào dùng mô hình nào, trade-offs là gì

---

## 🎓 Giảng viên AI: MIKU

> [!INFO] Về MIKU Agent
> **MIKU** là giảng viên AI chuyên về ML/DL/NLP, được thiết kế để hỗ trợ bạn học D2L theo phong cách **ELI5 (Explain Like I'm 5)** — giải thích đơn giản nhưng không làm mất tính chính xác kỹ thuật.

### 🌟 Phong cách giảng dạy của MIKU:

**Cấu trúc mỗi bài giảng:**

1. **🌱 Ví dụ đời thường** — Khởi đầu bằng ẩn dụ dễ hiểu
2. **🧠 Trực giác** — Giải thích cơ chế hoạt động
3. **📘 Định nghĩa kỹ thuật** — Khái niệm chính xác học thuật
4. **🧮 Công thức** — Toán học (nếu cần)
5. **🚀 Ứng dụng thực tế** — Dùng ở đâu, khi nào?
6. **⚖ Ưu – nhược điểm** — Trade-offs

**Nguyên tắc:**

- ✅ Luôn bắt đầu từ ví dụ đời thường trước
- ✅ Duy trì ẩn dụ nhất quán xuyên suốt bài giảng
- ✅ Giải thích "Tại sao?" trước khi đi vào "Như thế nào?"
- ✅ Không bao giờ nhảy thẳng vào công thức nếu chưa xây trực giác
- ✅ Mỗi khái niệm cần có: **Mục đích** (Why?) + **Cơ chế** (How?) + **Ứng dụng** (Where?)

### 📝 Quy trình đánh giá với MIKU:

Khi bạn hoàn thành một buổi học, nói với MIKU: **"Đã học xong [tên chủ đề]"**

MIKU sẽ:

1. **Tạo bài kiểm tra markdown** phù hợp với nội dung:
   - Buổi lý thuyết → Câu hỏi khái niệm + công thức
   - Buổi thực hành → Bài tập code/debug
   - Buổi kết hợp → Mix cả hai

2. **Đọc câu trả lời** và cho feedback:
   - Chỉ ra phần sai/thiếu ý
   - Giải thích lại theo ELI5
   - Nhấn mạnh điểm then chốt
   - **Không trừ điểm**, chỉ hướng dẫn cải thiện

**Format bài test chuẩn:**

```markdown
# 📝 Bài kiểm tra: [Tên chủ đề]

## Câu 1: Khái niệm cơ bản

[Giải thích khái niệm X bằng lời của bạn]

## Câu 2: Phân tích công thức

[Giải thích ý nghĩa từng thành phần trong công thức Y]

## Câu 3: So sánh

[So sánh A vs B, khi nào dùng cái nào?]

## Câu 4: Thực hành (nếu có)

[Implement/debug đoạn code hoặc trace output]
```

### 🔗 Tài liệu MIKU:

- **Agent Instructions:** `30_Resources/Books/Dive Into Deep Learning/d2l-en/AGENTS.md`
- Tham khảo file này để hiểu rõ hơn về phương pháp giảng dạy

---

## 📅 Timeline Tổng Quan

| Giai đoạn                    | Tuần  | Thời gian     | Nội dung chính                     |
| ---------------------------- | ----- | ------------- | ---------------------------------- |
| **Foundation**               | 1-2   | 02/03 - 15/03 | Setup + Preliminaries              |
| **Linear Models**            | 3-4   | 16/03 - 29/03 | Linear Regression + Classification |
| **Deep Feedforward**         | 5-6   | 30/03 - 12/04 | MLPs + Builders Guide              |
| **Computer Vision**          | 7-9   | 13/04 - 03/05 | CNNs (Traditional + Modern)        |
| **Sequential Models**        | 10-11 | 04/05 - 17/05 | RNNs + Modern RNNs                 |
| **Attention & Transformers** | 12-13 | 18/05 - 31/05 | Attention + Transformers (CORE!)   |
| **Advanced Topics**          | 14-15 | 01/06 - 14/06 | NLP/CV Applications + Optimization |
| **Specialization**           | 16    | 15/06 - 21/06 | Chọn 1: RL/GANs/GPs/RecSys         |
| **Review & Project**         | 17    | 22/06 - 01/07 | Ôn tập + Final Project             |

---

## 📖 Chi tiết từng tuần

### TUẦN 1: Foundation Setup (02/03 - 08/03)

**Mục tiêu:** Cài đặt môi trường, làm quen với workflow

#### Buổi 1 (02/03 - Chủ Nhật)

- **Tài liệu:** `chapter_installation/index.md`
- **Nội dung:**
  - Cài đặt Python environment (conda/mamba)
  - Clone repo d2l-en
  - Chạy thử notebook đầu tiên
  - Cài PyTorch/TensorFlow/JAX (chọn 1)
- **Output:** Environment hoạt động, chạy được `jupyter notebook`

#### Buổi 2 (04/03 - Thứ Ba)

- **Tài liệu:** `chapter_preliminaries/ndarray.md`
- **Nội dung:**
  - Tensor operations cơ bản
  - Broadcasting
  - Indexing & slicing
  - Memory efficiency
- **Bài tập:** Chạy toàn bộ code trong notebook

#### Buổi 3 (05/03 - Thứ Tư)

- **Tài liệu:** `chapter_preliminaries/linear-algebra.md`
- **Nội dung:**
  - Matrix operations
  - Norms
  - Eigenvalues/eigenvectors (khái niệm)
- **Bài tập:** Implement matrix multiply từ đầu

#### Buổi 4 (07/03 - Thứ Sáu)

- **Tài liệu:** `chapter_preliminaries/calculus.md`
- **Nội dung:**
  - Derivatives
  - Partial derivatives
  - Chain rule
  - Gradients
- **Bài tập:** Visualize gradient của hàm đơn giản

#### Buổi 5 (08/03 - Thứ Bảy)

- **Tài liệu:** `chapter_preliminaries/autograd.md`
- **Nội dung:**
  - Automatic differentiation
  - Computational graphs
  - Backward pass
- **Bài tập:** Tính gradient bằng autograd vs. tay

---

### TUẦN 2: Math Foundations (09/03 - 15/03)

**Mục tiêu:** Hoàn thành Preliminaries

#### Buổi 6 (09/03 - CN)

- **Tài liệu:** `chapter_preliminaries/probability.md`
- **Nội dung:**
  - Random variables
  - Distributions (Normal, Bernoulli, etc.)
  - Expectation & Variance
  - Bayes' theorem
- **Bài tập:** Visualize distributions

#### Buổi 7 (11/03 - Thứ Ba)

- **Tài liệu:** `chapter_preliminaries/pandas.md` + `lookup-api.md`
- **Nội dung:**
  - Data preprocessing với pandas
  - API documentation
- **Bài tập:** Load dataset và visualize

#### Buổi 8 (13/03 - Thứ Năm)

- **Tài liệu:** `chapter_introduction/index.md`
- **Nội dung:**
  - What is Deep Learning?
  - Key components
  - Historical context
  - Roadmap của sách
- **Bài tập:** Đọc và note key concepts

#### Buổi 9 (15/03 - Thứ Bảy)

- **REVIEW WEEK 1-2**
- **Mini Test:** Kiểm tra kiến thức Preliminaries
  - Questions về tensor ops, calculus, probability
  - Code challenge nhỏ

---

### TUẦN 3: Linear Regression (16/03 - 22/03)

**Mục tiêu:** Hiểu sâu Linear Regression (first ML model!)

#### Buổi 10 (16/03 - CN)

- **Tài liệu:** `chapter_linear-regression/linear-regression.md`
- **Nội dung:**
  - Problem formulation
  - Loss function (MSE)
  - Analytical solution (Normal Equation)
- **Bài tập:** Derive Normal Equation

#### Buổi 11 (18/03 - Thứ Ba)

- **Tài liệu:** `chapter_linear-regression/linear-regression-scratch.md`
- **Nội dung:**
  - Implement từ scratch
  - Gradient descent
  - Training loop
- **Bài tập:** Code Linear Regression không dùng library

#### Buổi 12 (20/03 - Thứ Năm)

- **Tài liệu:** `chapter_linear-regression/linear-regression-concise.md`
- **Nội dung:**
  - Dùng PyTorch/TF high-level API
  - `nn.Linear`, `nn.MSELoss`, `optim.SGD`
- **Bài tập:** So sánh scratch vs. concise

#### Buổi 13 (22/03 - Thứ Bảy)

- **Tài liệu:** `chapter_linear-regression/generalization.md`
- **Nội dung:**
  - Overfitting vs. Underfitting
  - Training/Validation/Test split
  - Regularization (L1/L2)
- **Bài tập:** Visualize overfitting

---

### TUẦN 4: Linear Classification (23/03 - 29/03)

**Mục tiêu:** Softmax Regression, multi-class classification

#### Buổi 14 (23/03 - CN)

- **Tài liệu:** `chapter_linear-classification/softmax-regression.md`
- **Nội dung:**
  - Softmax function
  - Cross-entropy loss
  - Why softmax? (vs. sigmoid)
- **Bài tập:** Derive cross-entropy gradient

#### Buổi 15 (25/03 - Thứ Ba)

- **Tài liệu:** `chapter_linear-classification/image-classification-dataset.md`
- **Nội dung:**
  - Fashion-MNIST dataset
  - Data loading
  - Data augmentation basics
- **Bài tập:** Load và visualize Fashion-MNIST

#### Buổi 16 (27/03 - Thứ Năm)

- **Tài liệu:** `chapter_linear-classification/softmax-regression-scratch.md`
- **Nội dung:**
  - Implement Softmax từ scratch
  - Training on Fashion-MNIST
- **Bài tập:** Achieve >80% accuracy

#### Buổi 17 (29/03 - Thứ Bảy)

- **Tài liệu:** `chapter_linear-classification/softmax-regression-concise.md`
- **Nội dung:**
  - High-level implementation
  - `nn.CrossEntropyLoss`
- **Bài tập:** Compare performance
- **REVIEW:** Mini test Linear Models

---

### TUẦN 5: Multilayer Perceptrons (30/03 - 05/04)

**Mục tiêu:** Hiểu sâu về hidden layers, activation functions, backprop

#### Buổi 18 (30/03 - CN)

- **Tài liệu:** `chapter_multilayer-perceptrons/mlp.md`
- **Nội dung:**
  - Why we need hidden layers?
  - Activation functions (ReLU, sigmoid, tanh)
  - Universal approximation theorem
- **Bài tập:** Visualize decision boundaries

#### Buổi 19 (01/04 - Thứ Ba)

- **Tài liệu:** `chapter_multilayer-perceptrons/mlp-implementation.md`
- **Nội dung:**
  - Implement MLP scratch + concise
  - Fashion-MNIST với MLP
- **Bài tập:** Achieve >85% accuracy

#### Buổi 20 (03/04 - Thứ Năm)

- **Tài liệu:** `chapter_multilayer-perceptrons/backprop.md`
- **Nội dung:**
  - Backpropagation algorithm
  - Chain rule in detail
  - Computational graph
- **Bài tập:** Manually calculate gradients cho 2-layer MLP

#### Buổi 21 (05/04 - Thứ Bảy)

- **Tài liệu:** `chapter_multilayer-perceptrons/numerical-stability-and-init.md`
- **Nội dung:**
  - Vanishing/exploding gradients
  - Weight initialization (Xavier, He)
  - Why initialization matters?
- **Bài tập:** Compare different init strategies

---

### TUẦN 6: Regularization & Deep Learning Builders (06/04 - 12/04)

#### Buổi 22 (06/04 - CN)

- **Tài liệu:** `chapter_multilayer-perceptrons/dropout.md`
- **Nội dung:**
  - Dropout mechanism
  - Why it works? (ensemble view)
  - Inverted dropout
- **Bài tập:** Implement dropout từ scratch

#### Buổi 23 (08/04 - Thứ Ba)

- **Tài liệu:** `chapter_multilayer-perceptrons/generalization-deep.md`
- **Nội dung:**
  - Generalization in deep learning
  - Bias-variance tradeoff
  - Early stopping
- **Bài tập:** Experiment với model capacity

#### Buổi 24 (10/04 - Thứ Năm)

- **Tài liệu:** `chapter_builders-guide/model-construction.md` + `parameters.md`
- **Nội dung:**
  - Blocks & Sequential
  - Parameter management
  - Custom layers
- **Bài tập:** Build custom MLP block

#### Buổi 25 (12/04 - Thứ Bảy)

- **Tài liệu:** `chapter_builders-guide/read-write.md` + `use-gpu.md`
- **Nội dung:**
  - Save/Load models
  - GPU training
- **Bài tập:** Train model trên GPU (nếu có)
- **KAGGLE PROJECT:** `kaggle-house-price.md` (Buổi thêm nếu có thời gian)

---

### TUẦN 7: Convolutional Neural Networks (13/04 - 19/04)

**Mục tiêu:** Hiểu sâu về convolution, pooling, và tại sao CNNs hoạt động

#### Buổi 26 (13/04 - CN)

- **Tài liệu:** `chapter_convolutional-neural-networks/conv-layer.md`
- **Nội dung:**
  - Convolution operation
  - Local connectivity
  - Translation equivariance
  - Padding & stride
- **Bài tập:** Implement conv2d từ scratch

#### Buổi 27 (15/04 - Thứ Ba)

- **Tài liệu:** `chapter_convolutional-neural-networks/pooling.md`
- **Nội dung:**
  - Max pooling vs. Average pooling
  - Why pooling? (translation invariance)
- **Bài tập:** Compare pooling strategies

#### Buổi 28 (17/04 - Thứ Năm)

- **Tài liệu:** `chapter_convolutional-neural-networks/lenet.md`
- **Nội dung:**
  - LeNet architecture
  - First successful CNN
  - Training on MNIST
- **Bài tập:** Implement và train LeNet

#### Buổi 29 (19/04 - Thứ Bảy)

- **Tài liệu:** Các sections khác trong chapter (channels, etc.)
- **Nội dung:**
  - Multiple input/output channels
  - 1x1 convolutions
- **Bài tập:** Experiment với channel sizes

---

### TUẦN 8: Modern CNNs - Part 1 (20/04 - 26/04)

**Mục tiêu:** Học các architectures quan trọng

#### Buổi 30 (20/04 - CN)

- **Tài liệu:** `chapter_convolutional-modern/alexnet.md`
- **Nội dung:**
  - AlexNet architecture
  - ReLU activation
  - Dropout in CNNs
  - GPU training breakthrough
- **Bài tập:** Train AlexNet trên Fashion-MNIST

#### Buổi 31 (22/04 - Thứ Ba)

- **Tài liệu:** `chapter_convolutional-modern/vgg.md`
- **Nội dung:**
  - VGG blocks
  - Depth matters
  - 3x3 conv stacking
- **Bài tập:** Build VGG-11

#### Buổi 32 (24/04 - Thứ Năm)

- **Tài liệu:** `chapter_convolutional-modern/nin.md`
- **Nội dung:**
  - Network in Network
  - 1x1 convolutions
  - Global average pooling
- **Bài tập:** Replace FC layers với GAP

#### Buổi 33 (26/04 - Thứ Bảy)

- **Tài liệu:** `chapter_convolutional-modern/googlenet.md`
- **Nội dung:**
  - Inception modules
  - Multi-scale features
  - Auxiliary classifiers
- **Bài tập:** Implement Inception block

---

### TUẦN 9: Modern CNNs - Part 2 (27/04 - 03/05)

#### Buổi 34 (27/04 - CN)

- **Tài liệu:** `chapter_convolutional-modern/batch-norm.md`
- **Nội dung:**
  - Batch Normalization (QUAN TRỌNG!)
  - Internal covariate shift
  - Training vs. inference mode
- **Bài tập:** Add BN vào LeNet, so sánh performance

#### Buổi 35 (29/04 - Thứ Ba)

- **Tài liệu:** `chapter_convolutional-modern/resnet.md`
- **Nội dung:**
  - Residual connections
  - Why ResNets work? (gradient flow)
  - Skip connections
- **Bài tập:** Implement ResNet block

#### Buổi 36 (01/05 - Thứ Năm) - **Nghỉ lễ, có thể học nhẹ nhàng**

- **Tài liệu:** `chapter_convolutional-modern/densenet.md`
- **Nội dung:**
  - Dense connections
  - Feature reuse
  - DenseNet vs. ResNet
- **Bài tập:** Compare parameter counts

#### Buổi 37 (03/05 - Thứ Bảy)

- **REVIEW CNNs**
- **Mini Test:** Architecture comparison
- **Project:** Train một modern CNN trên CIFAR-10

---

### TUẦN 10: Recurrent Neural Networks (04/05 - 10/05)

**Mục tiêu:** Hiểu sequential modeling

#### Buổi 38 (04/05 - CN)

- **Tài liệu:** `chapter_recurrent-neural-networks/sequence.md`
- **Nội dung:**
  - Sequence modeling basics
  - Autoregressive models
  - Markov models
- **Bài tập:** Implement simple sequence predictor

#### Buổi 39 (06/05 - Thứ Ba)

- **Tài liệu:** `chapter_recurrent-neural-networks/language-model.md`
- **Nội dung:**
  - Language modeling
  - Perplexity
  - N-gram models
- **Bài tập:** Train character-level LM

#### Buổi 40 (08/05 - Thứ Năm)

- **Tài liệu:** `chapter_recurrent-neural-networks/rnn.md`
- **Nội dung:**
  - RNN architecture
  - Hidden state
  - Backpropagation through time (BPTT)
- **Bài tập:** Implement vanilla RNN

#### Buổi 41 (10/05 - Thứ Bảy)

- **Tài liệu:** `chapter_recurrent-neural-networks/rnn-scratch.md` + `rnn-concise.md`
- **Nội dung:**
  - Implement RNN từ scratch
  - PyTorch RNN modules
- **Bài tập:** Train RNN language model

---

### TUẦN 11: Modern RNNs (11/05 - 17/05)

#### Buổi 42 (11/05 - CN)

- **Tài liệu:** `chapter_recurrent-modern/gru.md`
- **Nội dung:**
  - Gated Recurrent Units
  - Reset & update gates
  - Why gates? (gradient flow)
- **Bài tập:** Compare GRU vs. vanilla RNN

#### Buổi 43 (13/05 - Thứ Ba)

- **Tài liệu:** `chapter_recurrent-modern/lstm.md`
- **Nội dung:**
  - Long Short-Term Memory
  - Forget, input, output gates
  - Cell state vs. hidden state
- **Bài tập:** Implement LSTM từ scratch (challenging!)

#### Buổi 44 (15/05 - Thứ Năm)

- **Tài liệu:** `chapter_recurrent-modern/deep-rnn.md` + `bi-rnn.md`
- **Nội dung:**
  - Stacking RNNs
  - Bidirectional RNNs
- **Bài tập:** Build deep BiLSTM

#### Buổi 45 (17/05 - Thứ Bảy)

- **Tài liệu:** `chapter_recurrent-modern/seq2seq.md` + `encoder-decoder.md`
- **Nội dung:**
  - Encoder-Decoder architecture
  - Seq2Seq for translation
  - Context vector bottleneck
- **Bài tập:** Implement simple seq2seq
- **REVIEW RNNs**

---

### TUẦN 12: Attention Mechanisms (18/05 - 24/05)

**Mục tiêu:** Hiểu sâu Attention (QUAN TRỌNG NHẤT!)

#### Buổi 46 (18/05 - CN)

- **Tài liệu:** `chapter_attention-mechanisms-and-transformers/queries-keys-values.md`
- **Nội dung:**
  - Query-Key-Value paradigm
  - Attention as soft lookup
  - Attention scores
- **Bài tập:** Visualize attention với ví dụ đơn giản

#### Buổi 47 (20/05 - Thứ Ba)

- **Tài liệu:** `chapter_attention-mechanisms-and-transformers/attention-pooling.md`
- **Nội dung:**
  - Nadaraya-Watson kernel regression
  - Attention pooling
- **Bài tập:** Implement attention pooling

#### Buổi 48 (22/05 - Thứ Năm)

- **Tài liệu:** `chapter_attention-mechanisms-and-transformers/attention-scoring-functions.md`
- **Nội dung:**
  - Additive attention
  - Scaled dot-product attention
  - Comparison
- **Bài tập:** Implement cả 2 loại

#### Buổi 49 (24/05 - Thứ Bảy)

- **Tài liệu:** `chapter_attention-mechanisms-and-transformers/bahdanau-attention.md`
- **Nội dung:**
  - Bahdanau attention mechanism
  - Attention trong seq2seq
  - Alignment visualization
- **Bài tập:** Add attention vào seq2seq model

---

### TUẦN 13: Transformers (25/05 - 31/05)

**Mục tiêu:** Hiểu kiến trúc Transformer (core của modern NLP!)

#### Buổi 50 (25/05 - CN)

- **Tài liệu:** `chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.md`
- **Nội dung:**
  - Self-attention mechanism
  - Positional encoding
  - Why we need positional info?
- **Bài tập:** Implement self-attention

#### Buổi 51 (27/05 - Thứ Ba)

- **Tài liệu:** `chapter_attention-mechanisms-and-transformers/multihead-attention.md`
- **Nội dung:**
  - Multi-Head Attention
  - Why multiple heads?
  - Parallel attention computation
- **Bài tập:** Implement multi-head attention

#### Buổi 52 (29/05 - Thứ Năm)

- **Tài liệu:** `chapter_attention-mechanisms-and-transformers/transformer.md`
- **Nội dung:**
  - Full Transformer architecture
  - Encoder-Decoder với Attention
  - Layer normalization & residuals
  - Feed-forward networks
- **Bài tập:** Build Transformer từ scratch (lớn!)

#### Buổi 53 (31/05 - Thứ Bảy)

- **Tài liệu:** `chapter_attention-mechanisms-and-transformers/vision-transformer.md` + `large-pretraining-transformers.md`
- **Nội dung:**
  - Vision Transformer (ViT)
  - BERT, GPT concepts
  - Pretraining paradigms
- **Bài tập:** ViT trên small dataset
- **BIG REVIEW:** Attention & Transformers test

---

### TUẦN 14: Optimization & NLP Pretraining (01/06 - 07/06)

#### Buổi 54 (01/06 - CN)

- **Tài liệu:** `chapter_optimization/optimization-intro.md` + key sections
- **Nội dung:**
  - SGD variants (Momentum, Adam, RMSprop)
  - Learning rate scheduling
  - Convergence analysis
- **Bài tập:** Compare optimizers

#### Buổi 55 (03/06 - Thứ Ba)

- **Tài liệu:** `chapter_natural-language-processing-pretraining/word2vec.md`
- **Nội dung:**
  - Word embeddings
  - Skip-gram & CBOW
  - Negative sampling
- **Bài tập:** Train word2vec

#### Buổi 56 (05/06 - Thứ Năm)

- **Tài liệu:** `chapter_natural-language-processing-pretraining/bert.md`
- **Nội dung:**
  - BERT architecture
  - Masked language modeling
  - Next sentence prediction
- **Bài tập:** Fine-tune pretrained BERT

#### Buổi 57 (07/06 - Thứ Bảy)

- **Tài liệu:** `chapter_natural-language-processing-applications/sentiment-analysis.md`
- **Nội dung:**
  - Sentiment analysis
  - Text classification pipeline
- **Bài tập:** Build classifier với BERT

---

### TUẦN 15: Advanced Applications (08/06 - 14/06)

**Mục tiêu:** Chọn 1 trong các ứng dụng để đi sâu

#### Option A: Computer Vision Deep Dive

**Buổi 58-61:**

- Object detection (R-CNN, SSD, YOLO concepts)
- Semantic segmentation (FCN)
- Image augmentation techniques
- Bài tập: CIFAR-10 Kaggle competition

#### Option B: NLP Deep Dive

**Buổi 58-61:**

- Machine translation
- Question answering
- Natural language inference
- Bài tập: Fine-tune model cho specific task

#### Buổi 62 (14/06 - Thứ Bảy)

- **Tài liệu:** `chapter_computational-performance/` (selected topics)
- **Nội dung:**
  - Model parallelism
  - Async computation
  - Mixed precision training
- **Bài tập:** Optimize training speed

---

### TUẦN 16: Specialization (15/06 - 21/06)

**Mục tiêu:** Chọn 1 chủ đề nâng cao để học sâu

#### Option 1: Generative Models

- **Tài liệu:** `chapter_generative-adversarial-networks/`
- GANs (DCGAN, StyleGAN concepts)
- Mode collapse
- Training tricks

#### Option 2: Reinforcement Learning

- **Tài liệu:** `chapter_reinforcement-learning/`
- Q-learning
- Policy gradients
- Value-based vs. policy-based

#### Option 3: Recommender Systems

- **Tài liệu:** `chapter_recommender-systems/`
- Collaborative filtering
- Matrix factorization
- Deep recommender systems

#### Option 4: Gaussian Processes

- **Tài liệu:** `chapter_gaussian-processes/`
- Kernels
- GP regression
- Uncertainty quantification

**Deliverable:** Implement 1 project nhỏ với topic đã chọn

---

### TUẦN 17: Review & Final Project (22/06 - 01/07)

**Mục tiêu:** Củng cố toàn bộ kiến thức

#### Ngày 22-24/06: Review Marathon

- Review notes từ tuần 1-16
- Làm lại các bài tập khó
- Refactor code đã viết

#### Ngày 25-28/06: Final Project

**Chọn 1 trong các đề tài:**

1. **Image Classification:** Train SOTA model trên CIFAR-100/ImageNet subset
2. **Text Generation:** Build GPT-style model, fine-tune trên custom dataset
3. **Seq2Seq:** Machine translation hoặc summarization
4. **Multi-modal:** Image captioning (CNN + Transformer)

**Requirements:**

- Implement từ scratch hoặc fine-tune pretrained
- Write detailed report (5-10 trang)
- Visualizations & ablation studies

#### Ngày 29-30/06: Presentation Prep

- Tạo slides
- Demo video
- Viết blog post (optional)

#### Ngày 01/07: 🎉 HOÀN THÀNH!

- Self-assessment quiz
- Reflect on learning journey
- Plan next steps (papers to read, advanced topics)

---

## 📝 Quy tắc thực hành

### 🎯 Workflow mỗi buổi học (2-3 giờ):

#### Bước 1: Chuẩn bị (5 phút)

- Đọc file `progress.md` để nắm ngữ cảnh buổi học trước
- Xem lại notes đã tạo (nếu có)
- Mở file markdown + notebook tương ứng trong repo d2l-en

#### Bước 2: Học lý thuyết với MIKU (45-60 phút)

- **Đọc markdown file** từ repo d2l-en
- **Nếu gặp khái niệm khó hiểu:** Hỏi MIKU giải thích theo ELI5
  - Ví dụ: "_Giải thích backpropagation theo ELI5_"
  - MIKU sẽ trả lời theo format: Ví dụ đời thường → Trực giác → Định nghĩa → Công thức
- **Ghi chú quan trọng:**
  - Tạo **Concept Note** trong `20_Areas/AI/Concepts/` nếu là khái niệm mới
  - Tạo **Source Note** trong `30_Resources/Books/Dive Into Deep Learning/` cho chapter
  - Dùng cấu trúc ELI5 của MIKU trong notes

#### Bước 3: Thực hành code (60-75 phút)

- **Chạy Jupyter notebook** tương ứng
- **Đọc code từng cell:**
  - Nếu không hiểu → Hỏi MIKU: "_Giải thích đoạn code này làm gì_"
  - MIKU sẽ trace từng bước với comment
- **Thử nghiệm:**
  - Thay đổi hyperparameters
  - Visualize intermediate outputs
  - Break code và fix lại (học từ lỗi!)

#### Bước 4: Làm bài tập (30-45 phút)

- Hoàn thành exercises trong sách
- Nếu gặp khó khăn → Hỏi MIKU gợi ý (không hỏi đáp án trực tiếp)

#### Bước 5: Kiểm tra với MIKU (20-30 phút)

- **Nói với MIKU:** "_Đã học xong [tên chủ đề]_"
- **Nhận bài test** từ MIKU (3-5 câu hỏi)
- **Trả lời bằng markdown**
- **Nhận feedback** và sửa lại phần sai

#### Bước 6: Cập nhật progress (10 phút)

- Cập nhật `progress.md` theo format:
  ```yaml
  ---
  tác vụ: D2L - Tuần X, Buổi Y - [Tên chủ đề] - 2026-MM-DD
  nội dung: Đã hoàn thành [[Chapter_Name]]
  chi tiết:
    - Đã học: [khái niệm chính]
    - Đã code: [implementations]
    - Bài tập: [exercises hoàn thành]
    - Test score: [X/Y câu đúng]
    - Thời gian: [X giờ]
    - Ghi chú: [reflections/insights]
  ---
  ```

---

### 🧠 Sau mỗi chương lớn (Milestone Tests):

#### 1. Review Session

- Đọc lại tất cả concept notes đã tạo
- Làm lại exercises khó
- Visualize lại các công thức quan trọng

#### 2. ELI5 Summary

- **Challenge:** Giải thích toàn bộ chương cho "người 5 tuổi"
- Viết một bài blog/note ngắn (300-500 từ) theo cấu trúc:
  - Vấn đề chương này giải quyết
  - Giải pháp chính (intuition)
  - Key takeaways
- Đặt trong `30_Resources/Books/Dive Into Deep Learning/Summaries/`

#### 3. Big Test với MIKU

- Nói: "_Tạo bài kiểm tra tổng hợp cho [tên chương]_"
- MIKU sẽ tạo test 10-15 câu hỏi hỗn hợp
- Làm bài **không tra tài liệu** (active recall)
- Chấm và review với MIKU

#### 4. Code Refactor

- Viết lại code của chương theo style riêng
- Thêm comments chi tiết
- Push lên GitHub (optional nhưng khuyến khích)

---

### 💡 Tips học hiệu quả với MIKU:

**Cách đặt câu hỏi tốt:**

- ❌ Tránh: "_Gradient descent là gì?_" (quá chung chung)
- ✅ Tốt hơn: "_Giải thích gradient descent theo ELI5, tại sao phải dùng learning rate?_"
- ✅ Tốt nhất: "_Tôi hiểu gradient descent là đi xuống dốc, nhưng tại sao không nhảy thẳng tới điểm thấp nhất? Momentum giúp gì ở đây?_"

**Khi nào nên hỏi MIKU:**

- Khi gặp khái niệm mới và không rõ "big picture"
- Khi công thức toán phức tạp → Hỏi ý nghĩa từng thành phần
- Khi code có bug → Hỏi MIKU debug (nhưng tự trace trước)
- Khi muốn so sánh 2 approaches (VGG vs ResNet, RNN vs LSTM, etc.)

**Khi KHÔNG nên hỏi:**

- Câu hỏi có thể Google trong 10 giây (syntax error đơn giản)
- Hỏi đáp án bài tập thẳng (tự làm trước, gặp khó mới hỏi hint)

---

### 🚀 Chiến lược thành công:

- 🎯 **Consistency > Intensity:** Học đều 4-5 buổi/tuần (2-3 giờ/buổi) tốt hơn học dồn cuối tuần
- 🔄 **Active Recall:** Sau mỗi section, đóng sách lại, viết ra những gì nhớ được
- 🧪 **Experiment-Driven Learning:** Không chỉ chạy code, mà thay đổi nó để thấy điều gì xảy ra
- 🏗 **First Principles:** Luôn hỏi "Tại sao?" cho đến khi hiểu cơ bản nhất
- 📊 **Visualize Everything:** Plot loss curves, attention maps, activations, gradients
- 👥 **Community:** Join D2L Discord/Forum, nhưng tự suy nghĩ trước khi hỏi
- 💤 **Sleep on It:** Học concepts khó trước khi ngủ, não sẽ consolidate qua đêm

---

## 🎓 Kiểm tra tiến độ - Milestone Tests với MIKU

> [!TIP] Cách làm Milestone Test
>
> 1. Nói với MIKU: "_Tạo bài kiểm tra tổng hợp cho [chương/giai đoạn]_"
> 2. MIKU sẽ tạo test 10-15 câu hỏi hỗn hợp (lý thuyết + code)
> 3. Làm bài **không tra tài liệu** (tối đa 90 phút)
> 4. Submit câu trả lời cho MIKU
> 5. Nhận feedback chi tiết + điểm yếu cần cải thiện

### 📝 Test 1 (Tuần 2) - Preliminaries & Math Foundations

**Scope:** Tuần 1-2  
**Topics:**

- Tensor operations, broadcasting, indexing
- Linear algebra: matrix ops, norms, eigenvalues
- Calculus: derivatives, gradients, chain rule
- Automatic differentiation
- Probability: distributions, expectation, Bayes

**Yêu cầu MIKU tạo test:**

```
"Tạo bài kiểm tra tổng hợp cho Preliminaries (Chapter preliminaries).
Gồm 12 câu: 5 câu lý thuyết, 4 câu phân tích công thức, 3 bài code nhỏ."
```

**Expected outputs:**

- Implement linear regression từ đầu (only NumPy)
- Calculate gradients manually vs. autograd
- Visualize distributions

---

### 📝 Test 2 (Tuần 4) - Linear Models

**Scope:** Tuần 3-4  
**Topics:**

- Linear regression: MSE, gradient descent, normal equation
- Softmax regression: cross-entropy, multiclass classification
- Overfitting/underfitting, regularization (L1/L2)
- Fashion-MNIST dataset

**Yêu cầu MIKU:**

```
"Tạo bài kiểm tra cho Linear Regression + Classification.
Bao gồm so sánh MSE vs Cross-Entropy, khi nào dùng L1 vs L2."
```

**Project:** Classification trên Fashion-MNIST (>85% accuracy)

---

### 📝 Test 3 (Tuần 6) - MLPs & Deep Learning Builders

**Scope:** Tuần 5-6  
**Topics:**

- Activation functions: ReLU, sigmoid, tanh
- Backpropagation algorithm
- Vanishing/exploding gradients
- Weight initialization (Xavier, He)
- Dropout, early stopping
- Custom layers, parameter management

**Yêu cầu MIKU:**

```
"Tạo test cho MLPs. Nhấn mạnh câu hỏi về backprop (trace gradient flow)
và so sánh các activation functions."
```

**Code challenge:** Build MLP từ scratch với custom dropout layer

---

### 📝 Test 4 (Tuần 9) - Convolutional Neural Networks

**Scope:** Tuần 7-9  
**Topics:**

- Convolution operation: padding, stride, receptive field
- Pooling: max vs average
- Architectures: LeNet, AlexNet, VGG, GoogLeNet, ResNet, DenseNet
- Batch Normalization
- Why CNNs work? (translation equivariance/invariance)

**Yêu cầu MIKU:**

```
"Tạo test tổng hợp cho CNNs. Bao gồm:
1. Tính output size của conv layer với padding/stride
2. So sánh VGG vs ResNet (depth, parameters, gradient flow)
3. Giải thích tại sao batch norm giúp training nhanh hơn
4. Code: implement residual block"
```

**Project:** Train ResNet-18 trên CIFAR-10 (>90% accuracy)

---

### 📝 Test 5 (Tuần 11) - Recurrent Neural Networks

**Scope:** Tuần 10-11  
**Topics:**

- Sequence modeling, autoregressive models
- RNN architecture, hidden state
- BPTT (Backpropagation Through Time)
- Vanishing gradients trong RNNs
- GRU và LSTM: gates mechanism
- Bidirectional RNNs, deep RNNs
- Seq2Seq, encoder-decoder

**Yêu cầu MIKU:**

```
"Test RNNs: Nhấn mạnh sự khác biệt vanilla RNN vs LSTM vs GRU.
Tại sao LSTM giải quyết vanishing gradient? Cell state vs hidden state?
Code: implement LSTM cell từ scratch."
```

**Project:** Character-level language model (text generation)

---

### 📝 Test 6 (Tuần 13) - Attention & Transformers ⭐

**Scope:** Tuần 12-13 (QUAN TRỌNG NHẤT!)  
**Topics:**

- Query-Key-Value paradigm
- Attention scoring functions (additive vs scaled dot-product)
- Bahdanau attention
- Self-attention mechanism
- Multi-head attention
- Positional encoding
- Transformer architecture: encoder-decoder, layer norm, feed-forward
- Vision Transformer (ViT)

**Yêu cầu MIKU:**

```
"Tạo test KHÚC MẮC cho Attention & Transformers:
1. Tại sao self-attention tốt hơn RNN cho long sequences?
2. Giải thích scaled dot-product (tại sao chia cho sqrt(d)?)
3. Multi-head attention khác gì với single-head?
4. Positional encoding: tại sao dùng sin/cos?
5. Code: implement multi-head attention từ scratch"
```

**Project:** Transformer cho machine translation (small dataset)

---

### 📝 Test 7 (Tuần 15) - Advanced Topics

**Scope:** Tuần 14-15  
**Topics:**

- Optimizers: SGD, Momentum, Adam, RMSprop
- Learning rate scheduling
- Word2Vec: Skip-gram, CBOW, negative sampling
- BERT: MLM, NSP
- Fine-tuning pretrained models
- Model compression techniques

**Yêu cầu MIKU:**

```
"Test cho Optimization + NLP Pretraining.
So sánh Adam vs SGD+Momentum. BERT khác GPT như thế nào?"
```

---

### 🎯 Final Exam (Tuần 17) - Comprehensive

**Scope:** Toàn bộ 17 tuần  
**Format:**

- **Part 1 (90 phút):** 25 câu hỏi lý thuyết + phân tích
  - Cover tất cả chapters từ Preliminaries → Advanced
  - Mix: multiple choice, short answer, essay
- **Part 2 (3-4 ngày):** Final Project (xem section riêng)
- **Part 3 (30 phút):** Presentation + Q&A

**Yêu cầu MIKU:**

```
"Tạo Final Exam cho toàn bộ khóa D2L. Bao gồm:
- 10 câu foundational (linear models → MLPs)
- 8 câu architectures (CNNs, RNNs, Transformers)
- 7 câu advanced (optimization, transfer learning, trade-offs)
Focus vào 'big picture' và so sánh các approaches."
```

**Completion criteria:**

- Part 1: ≥70% correct
- Part 2: Working implementation + detailed report
- Part 3: Clear explanation of design choices

---

### 📊 Tracking Progress

**Tạo file:** `10_Projects/Đồ Án/D2L Test Results.md`

```markdown
# D2L Milestone Test Results

## Test 1 - Preliminaries (DD/MM/2026)

- Score: X/12
- Strengths: [...]
- Weaknesses: [...]
- MIKU Feedback: [...]

## Test 2 - Linear Models (DD/MM/2026)

...
```

---

## 📚 Tài liệu tham khảo

### Repo chính:

- **Repo:** `/home/sakana/Code/Obsidian-Vault/30_Resources/Books/Dive Into Deep Learning/d2l-en/`
- **Website:** https://d2l.ai/
- **Forum:** https://discuss.d2l.ai/

### Công cụ:

- **Notebook:** Jupyter Lab
- **Framework:** PyTorch (recommended) hoặc TensorFlow
- **Note-taking:** Obsidian (sử dụng vault hiện tại)
- **Tracking:** File này + `progress.md`

### Obsidian Integration:

- Tạo **Concept Notes** trong `20_Areas/AI/Concepts/` khi học khái niệm mới
- Tạo **Source Notes** trong `30_Resources/Books/Dive Into Deep Learning/` cho mỗi chapter
- Link concepts vào MOC: `20_Areas/AI/_MOCs/Deep Learning MOC.md`

---

## ✅ Checklist tổng quan

- [ ] Week 1-2: Foundation & Preliminaries
- [ ] Week 3-4: Linear Models
- [ ] Week 5-6: MLPs & Builders Guide
- [ ] Week 7-9: CNNs (Traditional + Modern)
- [ ] Week 10-11: RNNs (Vanilla + Modern)
- [ ] Week 12-13: Attention & Transformers
- [ ] Week 14: Optimization & NLP Pretraining
- [ ] Week 15: Advanced Applications
- [ ] Week 16: Specialization
- [ ] Week 17: Review & Final Project
- [ ] 🎓 Final Project Completed

---

## 🔄 Cập nhật progress

Sau mỗi buổi học, cập nhật vào `progress.md`:

```yaml
---
tác vụ: D2L Learning - Week X, Buổi Y - {{date}}
nội dung: Đã hoàn thành [[Chapter_Name]]
chi tiết:
  - Đã học: [concepts]
  - Đã code: [implementations]
  - Bài tập: [completed exercises]
  - Thời gian: [X giờ]
  - Ghi chú: [reflections]
---
```

---

> [!SUCCESS] Lời khuyên cuối
> Lộ trình này đầy đủ nhưng cũng linh hoạt. Nếu tuần nào bạn bận, có thể học chậm lại. Quan trọng là **hiểu sâu** hơn là chạy deadlines. Chúc bạn học tốt! 🚀

---

## 📞 Làm việc với MIKU - Quick Reference

### Các lệnh thường dùng:

```markdown
# Bắt đầu buổi học

"MIKU, hôm nay tôi học [tên chapter]. Giới thiệu sơ lược được không?"

# Khi gặp khái niệm khó

"Giải thích [concept] theo ELI5. Tại sao cần nó? Dùng khi nào?"

# Khi không hiểu công thức

"Giải thích từng thành phần của công thức [X]. Ý nghĩa vật lý/trực giác là gì?"

# Khi muốn so sánh

"So sánh [A] vs [B]. Khi nào dùng cái nào? Trade-offs?"

# Debug code

"Đoạn code này bị lỗi [paste code + error]. Giải thích tại sao và cách fix?"

# Sau khi học xong

"Đã học xong [chapter/section name]. Tạo bài kiểm tra cho tôi."

# Review trước test lớn

"Tạo bài kiểm tra tổng hợp cho [Chapter X - Chapter Y]."
```

### MIKU sẽ TỰ ĐỘNG:

- Giải thích theo cấu trúc ELI5 → Technical
- Dùng ví dụ đời thường trước khi đi vào toán
- Nhấn mạnh "Tại sao?" trước "Như thế nào?"
- Không trừ điểm khi bạn sai, chỉ giải thích lại

---

**Last updated:** 02/03/2026  
**Next review:** 15/03/2026 (sau tuần 2)  
**MIKU Agent:** `30_Resources/Books/Dive Into Deep Learning/d2l-en/AGENTS.md`
