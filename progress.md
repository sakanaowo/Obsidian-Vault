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
tác vụ: Viết lại MAE Paper Note (Comprehensive Rewrite) - 2026-01-14
nội dung: Đã viết lại hoàn toàn [[Masked Autoencoders Are Scalable Vision Learners]] theo chuẩn "First Principles" với từng đoạn đối chứng văn bản gốc.
chi tiết:
  - Cấu trúc mới: Mỗi phần bao gồm (1) Quote gốc tiếng Anh, (2) Dịch nghĩa Tiếng Việt, (3) Giải thích sâu/phân tích.
  - Đã thêm các bảng kết quả ablation (Masking Ratio, Decoder Design, Mask Token, Reconstruction Target, Data Augmentation, Mask Sampling Strategy).
  - Đã tích hợp hình ảnh minh họa từ `assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/` (Figure 1 architecture, reconstruction examples, mask sampling strategies).
  - Đã thêm công thức toán học để giải thích cơ chế (attention complexity, MSE loss, masking ratio effect).
  - Đã liên kết tới các Concept Notes: [[Masked Autoencoders (MAE)]], [[Vision Transformers (ViT)]], [[Self-Supervised Learning (Computer Vision)]], [[BERT]], [[Contrastive Learning]].
---
