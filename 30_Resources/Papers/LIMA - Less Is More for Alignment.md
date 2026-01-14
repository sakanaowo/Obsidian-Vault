---
title: "LIMA: Less Is More for Alignment"
aliases:
  - LIMA paper
  - Less Is More for Alignment
  - arXiv 2305.11206
type: source-note
source: arXiv
arxiv: 2305.11206v1
year: 2023
authors:
  - Chunting Zhou
  - Yuning Mao
  - Pengfei Liu
  - Xuezhe Ma
  - Gargi Ghosh
  - Puxin Xu
  - Avia Efrat
  - Mike Lewis
  - Srini Iyer
  - Ping Yu
  - Lili Yu
  - Luke Zettlemoyer
  - Jiao Sun
  - Susan Zhang
  - Omer Levy
pdf: assets/Library/2305.11206v1.pdf
tags:
  - paper
  - llm
  - alignment
  - instruction-tuning
  - sft
---

# Mục tiêu của bài báo (vấn đề gốc)

Hầu hết hệ thống **alignment** cho LLM hiện đại được mô tả như một “giai đoạn 2” cực nặng: thu thập hàng chục nghìn tới hàng triệu cặp *instruction–response* và/hoặc chạy [[Reinforcement Learning from Human Feedback (RLHF)]] để “dạy” mô hình trở nên hữu ích, lịch sự, an toàn. Bài báo này đặt một câu hỏi mang tính kiểm chứng: nếu mô hình nền đã đủ mạnh ở giai đoạn **pretraining**, vậy “alignment” thực sự cần nhiều dữ liệu và kỹ thuật phức tạp đến mức nào, hay phần lớn chỉ là dạy mô hình *cách trình bày* và *cách vào vai trợ lý*?

Từ đó họ đề xuất một khung diễn giải có tên [[Superficial Alignment Hypothesis]]: kiến thức và năng lực cốt lõi đã được học gần như toàn bộ trong pretraining; alignment chủ yếu học “phân phối bề mặt” của định dạng/giọng điệu khi tương tác với người dùng.

> [!NOTE] Ý nghĩa thực nghiệm
> Nếu giả thuyết đúng, ta kỳ vọng một mô hình nền mạnh có thể “mở khóa” hành vi trợ lý chất lượng cao chỉ bằng [[Instruction Tuning]] kiểu [[Supervised Fine-Tuning (SFT)]], với một tập ví dụ nhỏ nhưng được tuyển chọn kỹ.

# Tóm tắt (dịch từ Abstract)

Các mô hình ngôn ngữ lớn thường được huấn luyện qua hai giai đoạn: (1) **pretraining** không giám sát trên văn bản thô để học các biểu diễn tổng quát; và (2) **instruction tuning** quy mô lớn cùng học tăng cường để căn chỉnh với tác vụ và sở thích người dùng. Chúng tôi đo tầm quan trọng tương đối của hai giai đoạn này bằng cách huấn luyện **LIMA**, một mô hình LLaMa 65B tham số, được fine-tune bằng loss giám sát chuẩn trên chỉ **1.000** prompt–response được tuyển chọn cẩn thận, không dùng RLHF hay mô hình hóa preference. LIMA thể hiện hiệu năng mạnh đáng kể, học được cách tuân theo các định dạng trả lời cụ thể chỉ từ vài ví dụ, bao gồm các truy vấn phức tạp như lập kế hoạch du lịch hay suy đoán lịch sử giả định. Mô hình cũng có xu hướng tổng quát hóa tốt cho các tác vụ chưa từng xuất hiện trong tập huấn luyện. Trong một nghiên cứu so sánh bằng đánh giá con người có kiểm soát, câu trả lời của LIMA được xem là tương đương hoặc được ưu tiên hơn GPT-4 trong 43% trường hợp; con số này đạt 58% khi so với Bard và 65% khi so với DaVinci003 (được huấn luyện với phản hồi người dùng). Tổng hợp lại, các kết quả gợi ý mạnh rằng gần như toàn bộ kiến thức trong LLM được học trong pretraining, và chỉ cần lượng dữ liệu instruction tuning hạn chế để dạy mô hình tạo ra đầu ra chất lượng cao.

# Dữ liệu alignment: “ít nhưng tuyển chọn”

LIMA dùng đúng 1.000 ví dụ (khoảng 750k token). Điểm quan trọng không phải “ít”, mà là “ít nhưng có cấu trúc sư phạm”: đầu vào đa dạng, đầu ra cùng một **phong cách trợ lý hữu ích** (style consistency). Nguồn dữ liệu gồm các cặp hỏi–đáp chất lượng cao từ Stack Exchange (chia STEM và non-STEM), wikiHow, một phần WritingPrompts; kèm thêm ví dụ tự viết để tối ưu độ phủ tác vụ và “đồng bộ giọng điệu”. Bài báo cũng chủ động đưa vào 13 ví dụ có yếu tố độc hại/malevolent để kiểm tra khả năng từ chối và giải thích lý do từ chối (một “mồi” an toàn rất nhỏ, nhưng có chủ đích).

Điểm đáng chú ý ở đây là họ không “mua” dữ liệu bằng số lượng. Họ mua bằng ba thứ: (i) prompt phải tự-contained để mô hình không cần ngữ cảnh ngoài; (ii) response đủ dài và giàu lý giải để thể hiện chuẩn trợ lý; (iii) lọc bỏ các dấu vết “tôi/ý kiến cá nhân”, tham chiếu “như câu trả lời khác…”, liên kết/HTML… nhằm giảm nhiễu phong cách và tránh làm mô hình học thói quen trả lời kiểu diễn đàn.

# Huấn luyện: SFT chuẩn, tối ưu cho hội thoại

LIMA xuất phát từ LLaMa 65B và fine-tune theo [[Supervised Fine-Tuning (SFT)]]. Một chi tiết kỹ thuật nhỏ nhưng “đúng bản chất” cho alignment là họ thêm token kết thúc lượt (EOT) để phân tách vai “user/assistant”, tránh việc dùng EOS vốn có thể mang nghĩa khác trong mô hình nền. Họ fine-tune 15 epoch với AdamW, sequence length 2048 (cắt bớt nếu dài), và dùng residual dropout tăng dần theo tầng; checkpoint được chọn thủ công dựa trên dev set 50 ví dụ vì perplexity không tương quan tốt với chất lượng sinh.

> [!NOTE] Cách đọc kết quả
> Nếu alignment thực chất là “học phân phối định dạng”, thì việc thêm EOT và ép role-format ổn định đóng vai trò như một “mỏ neo” để mô hình kích hoạt đúng chế độ hành vi đã tiềm ẩn từ pretraining.

# Đánh giá: preference study + phân tích tuyệt đối

Thiết kế đánh giá chính của bài báo là so sánh theo **human preference** trên 300 prompt khó, với các baseline gồm Alpaca 65B (52k ví dụ SFT), DaVinci003 (RLHF), Bard, Claude và GPT-4. Kết quả mô tả theo hướng “LIMA được xem là tốt hơn hoặc ngang” ở một tỷ lệ đáng kể: 65% so với DaVinci003, 58% so với Bard, và 43% so với GPT-4 (Claude khoảng 46% theo mô tả trong bài). Nhóm tác giả còn lặp lại quy trình bằng cách dùng GPT-4 làm annotator và quan sát xu hướng tương tự; đồng thời họ báo cáo mức đồng thuận giữa người–người và người–GPT-4 ở mức cao (xấp xỉ 78–82% tùy cặp), như một lập luận rằng việc dùng LLM làm người chấm có thể phản ánh khá sát thói quen đánh giá của crowd.

Ngoài so sánh tương đối, họ làm “đánh giá tuyệt đối” trên 50 mẫu ngẫu nhiên: phân loại Fail/Pass/Excellent. Trong mẫu này, khoảng một nửa được coi là Excellent và chỉ một số ít bị Fail. Với safety, khi thử 30 prompt nhạy cảm, LIMA phản hồi “an toàn” khoảng 80% (nhưng vẫn có trường hợp nguy hiểm khi ý đồ xấu ẩn).

# Ablations: vì sao “less is more”?

Phần thuyết phục nhất của LIMA không nằm ở con số 1.000, mà nằm ở ablation tách ba trục: **đa dạng prompt**, **chất lượng response**, và **số lượng**. Trên một thiết lập nhỏ hơn (LLaMa 7B), họ cho thấy dữ liệu đa dạng (Stack Exchange) tốt hơn dữ liệu đồng chất kiểu “how-to” (wikiHow) khi giữ chất lượng cao; dữ liệu đã lọc chất lượng tốt hơn dữ liệu thô; và đặc biệt, tăng số lượng theo cấp số nhân từ một nguồn (chỉ tăng quantity) sớm bị “plateau” theo thang điểm hữu ích do GPT-3.5 chấm. Diễn giải nhất quán với [[Superficial Alignment Hypothesis]] là: alignment không ăn “token thừa”, nó ăn “tín hiệu định dạng + phủ tác vụ”, nên quantity mà không tăng diversity/quality dễ trở thành dữ liệu lặp/nhạt.

# Multi-turn: có thật sự “biết chat”?

Điểm dễ bị nghi ngờ là: fine-tune trên 1.000 lượt đơn liệu có sinh được hội thoại nhiều lượt? Họ thử 10 cuộc hội thoại thực; mô hình zero-shot vẫn có lúc mạch lạc, nhưng thường trật nhịp sau vài lượt. Khi thêm đúng 30 chuỗi multi-turn (tổng 1.030 ví dụ), chất lượng hội thoại tăng mạnh (tỷ lệ Excellent tăng rõ rệt; số lỗi giảm mạnh). Kết luận thực nghiệm ở đây là: năng lực “giữ ngữ cảnh” có thể đã nằm trong mô hình nền, nhưng cần một lượng ví dụ rất nhỏ để “định tuyến” mô hình vào đúng chế độ hội thoại.

# Hàm ý và cách dùng thực tế

Nếu bạn đang huấn luyện một mô hình nền mạnh cho trợ lý, LIMA gợi ý một chiến lược thực dụng: đầu tư vào [[Data Curation]] thay vì đốt compute cho hàng trăm nghìn ví dụ. “Alignment data” nên được nhìn như một “chuẩn giao diện”: nó dạy mô hình cách trình bày, cách từ chối, cách giữ giọng điệu nhất quán, và cách bộc lộ tri thức đã có. Tuy nhiên, safety là điểm LIMA tự thừa nhận còn yếu: một “mồi” 13 ví dụ từ chối không thể thay thế hệ thống an toàn toàn diện, và kết quả 80% safe trên một tập nhỏ không phải là bằng chứng đủ mạnh cho triển khai rủi ro cao.

> [!NOTE] Suy luận thêm (không phải phát biểu trực tiếp của bài)
> Có thể xem instruction tuning như “đặt mặt nạ giao tiếp” lên mô hình nền: nó không tạo thêm tri thức mới một cách đáng kể, nhưng thay đổi mạnh phân phối hành vi quan sát được. Điều này giải thích vì sao những mô hình “biết nhiều nhưng nói dở” có thể cải thiện nhanh khi được dạy đúng format, và cũng giải thích vì sao safety/harmful behavior không tự biến mất nếu mô hình nền từng thấy nhiều mẫu nguy hiểm trong pretraining.

# Giới hạn (đọc phê bình)

Bài báo dựa vào một mô hình nền rất mạnh (LLaMa 65B), nên kết luận “1.000 ví dụ là đủ” không nên bị hiểu sai thành quy tắc phổ quát cho mọi quy mô; với mô hình nhỏ hơn, tín hiệu alignment có thể cần nhiều hơn để ổn định. Thứ hai, việc “curate 1.000 ví dụ” thực chất là một dự án tri thức: nó đòi hiểu rõ mục tiêu trợ lý, đánh đổi giữa diversity và style, và xử lý các trường hợp edge-case. Thứ ba, preference study là thước đo quan trọng nhưng vẫn chủ quan, phụ thuộc vào hướng dẫn chấm và tập prompt; do đó, phần kết luận mạnh nhất nên đọc như: “chất lượng dữ liệu alignment có thể quan trọng hơn số lượng”, thay vì “RLHF là thừa”.

