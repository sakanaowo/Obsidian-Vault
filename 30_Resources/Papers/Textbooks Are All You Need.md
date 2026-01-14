---
title: "Textbooks Are All You Need"
aliases:
  - phi-1 paper
  - arXiv 2306.11644
type: source-note
source: arXiv
arxiv: 2306.11644v2
year: 2023
authors:
  - Suriya Gunasekar
  - Yi Zhang
  - Jyoti Aneja
  - Caio César Teodoro Mendes
  - Allie Del Giorno
  - Sivakanth Gopi
  - Mojan Javaheripi
  - Piero Kauffmann
  - Gustavo de Rosa
  - Olli Saarikivi
  - Adil Salim
  - Shital Shah
  - Harkirat Singh Behl
  - Xin Wang
  - Sébastien Bubeck
  - Ronen Eldan
  - Adam Tauman Kalai
  - Yin Tat Lee
  - Yuanzhi Li
pdf: assets/Library/2306.11644v2.pdf
tags:
  - paper
  - llm
  - code
  - data-quality
  - synthetic-data
---

# Câu hỏi trung tâm: “scale” không chỉ là tham số và compute

Trong nhiều năm, câu chuyện thống trị về LLM là **scaling laws**: tăng tham số hoặc compute thì hiệu năng tăng tương đối dự đoán được. Bài báo này thử đẩy một trục khác: **chất lượng dữ liệu**. Lập luận cốt lõi rất “first principles”: nếu mục tiêu là học ánh xạ *ngôn ngữ tự nhiên → mã nguồn*, thì dữ liệu tối ưu phải giống một cuốn giáo trình tốt: tự-contained, có giải thích, cân bằng khái niệm, và tránh rác/boilerplate. Dữ liệu kiểu “đống repo code” có thể rất lớn nhưng chứa nhiều mẫu không sư phạm (không dạy được lập luận thuật toán).

# Tóm tắt (dịch từ Abstract)

Chúng tôi giới thiệu **phi-1**, một mô hình ngôn ngữ lớn mới cho lập trình với kích thước nhỏ hơn đáng kể so với các mô hình cạnh tranh: phi-1 là một Transformer 1,3B tham số, được huấn luyện 4 ngày trên 8 A100, sử dụng một tuyển chọn dữ liệu “chất lượng giáo trình” từ web (6B token) và các “giáo trình + bài tập” tổng hợp bằng GPT-3.5 (1B token). Dù quy mô nhỏ, phi-1 đạt pass@1 50,6% trên HumanEval và 55,5% trên MBPP. Mô hình cũng thể hiện các đặc tính “emergent” đáng ngạc nhiên so với phi-1-base (trước giai đoạn fine-tune trên bài tập lập trình) và phi-1-small (350M tham số, huấn luyện cùng pipeline), trong đó phi-1-small vẫn đạt 45% trên HumanEval.

# Ba thành phần dữ liệu: CodeTextbook và CodeExercises

Pipeline của họ rất rõ ràng: trước tiên tạo một mô hình nền cho “ngôn ngữ + code” nhưng được làm sạch theo tiêu chí sư phạm, gọi là **CodeTextbook** (tổng <7B token khi gộp các phần); sau đó dùng một tập bài tập dạng “điền hàm theo docstring” để căn chỉnh hành vi sinh code, gọi là **CodeExercises**.

Thành phần thứ nhất là một tập code-language lọc từ The Stack và StackOverflow. Họ bắt đầu với >35B token Python, rồi dùng GPT-4 để gán nhãn *giá trị giáo dục* cho khoảng 100k mẫu (mục tiêu: “hữu ích cho học sinh học các khái niệm coding cơ bản”). Từ đó, họ huấn luyện một bộ phân loại (random forest trên embedding) để lọc ra khoảng 6B token “đáng học”. Bản chất của bước này là [[Data Curation]]: không phải “ít token”, mà là “ít nhiễu”, tức tăng tỷ lệ tín hiệu về cấu trúc, logic và giải thích.

Thành phần thứ hai là một “giáo trình tổng hợp” <1B token do GPT-3.5 sinh, trộn giữa văn bản giải thích và đoạn code minh họa, chủ đích nhắm các chủ đề thúc đẩy lập luận và kỹ năng thuật toán. Bài báo nhấn mạnh khó khăn thật sự của [[Synthetic Data (LLM Training)]]: nếu prompt không có cơ chế tạo đa dạng, mô hình sinh sẽ tạo ra dữ liệu đồng dạng và lặp. Họ lấy cảm hứng từ TinyStories (tiêm ràng buộc ngẫu nhiên vào prompt) để buộc đầu ra đa dạng nhưng vẫn giữ chất lượng.

Cuối cùng, **CodeExercises** (~180M token) là tập bài tập dạng docstring + lời giải để fine-tune: mục tiêu trực tiếp là huấn luyện hành vi “nhìn mô tả → hoàn thiện hàm Python”. Điều đáng chú ý là họ coi fine-tune trên CodeExercises không chỉ để tăng điểm benchmark, mà còn như một “tác nhân tái tổ chức tri thức”: sau fine-tune, mô hình làm tốt hơn cả những việc không xuất hiện trong CodeExercises (ví dụ dùng thư viện ngoài).

# Kiến trúc và huấn luyện (các con số quan trọng)

phi-1 có 1,3B tham số (24 layer, hidden 2048, MLP 8192, 32 head), sequence length 2048; phi-1-small 350M tham số (20 layer, hidden 1024, MLP 4096, 16 head). Họ pretrain phi-1-base trên CodeTextbook khoảng 8 epoch (tương đương >50B token được “nhìn thấy”), sau đó fine-tune để ra phi-1 trong vài nghìn bước. Điều họ muốn người đọc rút ra không phải “một cấu hình thần kỳ”, mà là: với dữ liệu có tính “giáo trình”, mô hình nhỏ vẫn có thể học hiệu quả hơn rất nhiều so với dữ liệu khổng lồ nhưng nhiều rác.

# Kết quả và diễn giải: “data quality bẻ cong scaling laws”

Kết quả headline là pass@1 50,6% HumanEval và 55,5% MBPP với 1,3B tham số và chỉ ~7B token dữ liệu gốc. Bài báo còn chỉ ra một mốc trung gian quan trọng: phi-1-base (chưa fine-tune bài tập) đã đạt ~29% HumanEval, và việc fine-tune trên CodeExercises kéo hiệu năng lên mạnh, đồng thời mở khóa năng lực mới.

> [!NOTE] Tại sao một tập bài tập nhỏ có thể “mở khóa”?
> Bài báo mô tả (và minh họa bằng ví dụ) rằng fine-tuning có thể cải thiện “khả năng tuân thủ và hiểu yêu cầu” hơn là chỉ học thêm cú pháp. Một cách đọc hợp lý là: pretraining tích lũy mảnh tri thức rời rạc; fine-tuning trên bài tập dạng docstring buộc mô hình học *cách kết nối* các mảnh đó thành chuỗi hành động giải bài (planning nhỏ), nên lợi ích lan sang tác vụ ngoài phân phối.

# Ví dụ định tính (những gì paper muốn bạn “thấy tận mắt”)

Trong Section 3, họ đối chiếu phi-1, phi-1-base và phi-1-small trên các prompt tự tạo. Một ví dụ cho thấy phi-1-base “lạc đề” khi xử lý quan hệ logic (chọn số, chia hết, mô phỏng điểm), trong khi phi-1 sau fine-tune có thể tạo vòng lặp, lấy random theo ràng buộc, và tính điểm đúng. Một ví dụ khác là dùng PyGame/Tkinter: dù CodeExercises chủ yếu dùng thư viện Python cơ bản, phi-1 sau fine-tune thể hiện khả năng gọi API ngoài đúng logic hơn; phi-1-base có thể nhớ vài lệnh nhưng thường sai ngữ nghĩa; phi-1-small hiểu logic phần nào nhưng thiếu dung lượng để gắn đúng API.

# Decontamination (tránh “học tủ” benchmark)

Bài báo dành một phần để bàn về khả năng “contamination” (dữ liệu huấn luyện vô tình chứa bài tương tự HumanEval). Họ mô tả việc lọc/prune tập CodeExercises theo ngưỡng tương đồng và kiểm tra hiệu năng theo hai nhóm bài “similar” và “non-similar”, rồi lập luận rằng ngay cả khi prune mạnh, phi-1 vẫn giữ lợi thế lớn so với baseline được nhắc (ví dụ StarCoder-Prompted), nên cải thiện không thể quy về học tủ đơn giản. Đây là điểm quan trọng vì toàn bộ luận đề “data quality” sẽ sụp nếu kết quả chủ yếu đến từ rò rỉ benchmark.

# Hàm ý thực hành

Điểm thực dụng của bài báo là mở ra một lộ trình huấn luyện mô hình code nhỏ/nhẹ: thay vì đua “tham số × token”, ta thiết kế dữ liệu có cấu trúc sư phạm và tối ưu cho ánh xạ *mô tả → chương trình*. Điều này có thể giảm chi phí môi trường và hạ rào cản tái lập. Nhưng bài báo cũng thẳng thắn rằng họ không công bố đầy đủ chi tiết sinh dữ liệu tổng hợp vì lý do sở hữu; do đó, một phần “bí quyết” nằm ở nghệ thuật prompt/curation hơn là kiến trúc.

> [!NOTE] Suy luận thêm (không phải phát biểu trực tiếp của bài)
> “Textbook-quality” có thể được hiểu như việc tối ưu *mutual information* giữa mô tả tự nhiên và đoạn code: dữ liệu tốt tăng độ nén có ý nghĩa (ít token nhưng mỗi token mang nhiều tín hiệu lập luận). Dưới góc nhìn này, scaling laws “theo token thô” là thước đo sai đơn vị khi token chứa quá nhiều nhiễu.

