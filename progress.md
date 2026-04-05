---
---

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
