---
aliases:
  - Self-Attention
tags:
  - Self-Attention
  - seq2seq
  - LSTM
  - RNN
  - Positional-Encoding
Reference:
  - "[[Transformer]]"
  - "[[Attention is all you need]]"
  - "[[Positional Encoding|Positional Encoding]]"
Requirement:
  - "[[Positional Encoding|Positional Encoding]]"
---
# Self-attention là gì ?
Như chúng ta đã biết, **Word Embedding** là vector đại diện cho ngữ nghĩa của một từ trong câu. Trong bước tiền xử lý, chúng ta đã tạo ra một **không gian vector** chứa các vector embedding của các từ. Những từ có nghĩa tương tự nhau sẽ có vector ở gần nhau trong không gian đó và ngược lại. Tuy nhiên, ý nghĩa của các từ riêng lẻ trong một câu không thể đại diện cho toàn bộ ý nghĩa của câu đó. Chẳng hạn, trong câu "The apple on the table", từ "apple" trong ngữ cảnh này được hiểu là quả "táo" . Nhưng nếu đặt trong một ngữ cảnh khác, như trong câu "The Apple keynote was interesting", từ "Apple" có thể ám chỉ đến công ty công nghệ.
Cơ chế Self-Attention được đề xuất trong bài báo [Attention Is All You Need](https://arxiv.org/abs/1706.03762v7) có thể giải quyết tốt vấn đề này. Ý tưởng của nó là so sánh các từ với nhau đôi một, bao gồm cả chính nó (self), để tìm ra mức độ quan trọng của mỗi từ mà mô hình nên chú ý tới (thể hiện qua trọng số). Điều này giúp mô hình hiểu đúng ý nghĩa của từ trong ngữ cảnh cụ thể, thay vì chỉ dựa vào ý nghĩa tổng quát của từ đó khi đứng riêng lẻ.

# Cơ chế hoạt động
## Cơ chế Attention trong seq2seq
Nếu chúng ta chỉ dịch từng từ ở ngôn ngữ này sang ngôn ngữ khác bằng cách ánh xạ word-by-word , thì đó quả là một phương pháp tệ, thiếu hiệu quả do không học được thông tin từ những từ xung quanh.

![](https://i.imgur.com/UHBHp9Q.png)

Mô hình `seq2seq` đã giới thiệu cơ chế `attention`. Trong cơ chế này, những từ đích được 'chú ý' với các từ trong câu nguồn nhằm xác định xem mối quan hệ.Mỗi từ sẽ được biểu diễn dưới dạng vector embedding và có nhiều cách để tính toán "**score attention**".
Ở đây, chúng ta sẽ sử dụng **tích vô hướng** (dot product). Hai từ có **ngữ nghĩa càng tương đồng** thì tích **vô hướng vector embedding của chúng càng lớn** (trong trường hợp này score attention càng cao ). Với mỗi từ đích , chúng ta tính tích vô hướng với tất cả các từ trong câu nguồn.
Sau khi tính toán, chúng ta thu được một vector chứa các "score attention". Kết quả này được đưa qua hàm softmax để chuẩn hóa, từ đó xác định từ đích nên "chú ý" bao nhiêu phần trăm đến các từ trong câu nguồn.

![](https://i.imgur.com/aMtugSb.png)

Giả sử dịch câu "**Tôi rất thích ăn cơm nếp**" sang tiếng anh và ta có vector embedding của các từ trên:
- `tôi: [-0.124, 0.067, -0.089]`,
- `rất: [0.156, -0.112, 0.078]`,
- `thích: [-0.082, 0.145, -0.167]`,
- `ăn: [0.134, -0.156, 0.112]`,
- `cơm: [-0.167, 0.089, -0.134]`,
- `nếp: [0.112, -0.145, 0.091]`,
- `s1: [0.23, 0.34, 0.45]`,

#### Score attention 
$$e^1=s_1(\begin{bmatrix}|&|&|&|&|&| \\ h_1&h_2&h_3&h_4&h_5&h_6 \\|&|&|&|&|&| \end{bmatrix})$$

Thay số => $=[−0.04579,0.0329,−0.04471,0.02818,−0.06845,0.01741]$

#### softmax 
$$\alpha^1=\frac{\exp(e^1)}{\sum_{i=1}^{d_{e^1}}\exp(e^1_i)}$$
$=[0.16122379,0.174423,0.161398,0.17360166,0.15761154,0.17174201]$

=> kết quả cho thấy `s1` chú ý tới các từ trong câu nguồn là gần như tương đương nhau 


> [!NOTE] Note
> attention trong `seq2seq` không tự chú ý đến chính nó vì `seq2seq` thường dùng **RNN**  hoặc **LSTM** cho encoder và decoder 


## Chi tiết về #Self-Attention 

Cơ chế #Self-Attention  và `attention` ở trên có chút khác biệt 
Self-attention **tự chú ý tới chính nó**

Xét câu: "**Tôi cảm thấy tôi không được khoẻ**"
Cơ chế self-attention giúp mô hình nhận ra sự liên quan giữa 2 lần xuất hiện của từ "tôi"
-> khi từ "Tôi" tự "chú ý" đến chính nó thì mô hình sẽ hiểu đây là chủ ngữ và thực hiện hành động đọc nhằm phân biệt với từ "tôi" thứ 2 không phải là chủ ngữ.
Lúc này việc **tự chú ý** đến nó giúp mô hình **hiểu ngữ cảnh một cách chính xác**. Ngoài ra nó còn giúp **giữ lại ngữ nghĩa cho từ**, việc tự chú ý cho phép mỗi từ giữ lại ngữ nghĩa riêng của nó.
Ví dụ: **"Tôi đọc sách của tôi."** 
	Từ "tôi" khi xuất hiện lần thứ hai sẽ "nhớ" lại rằng nó đại diện cho người sở hữu cuốn sách, trong khi vẫn hiểu rõ rằng "Tôi" đầu tiên là chủ thể thực hiện hành động đọc.


> [!NOTE] Tóm lại
> việc tự chú ý là cần thiết để mô hình **giữ được ngữ nghĩa nguyên bản** của từ và **phân biệt chính xác các mối quan hệ ngữ cảnh trong câu**, đặc biệt trong cấu trúc phức tạp và câu từ có lặp lại 


### Positional Encoding

**Transformers xử lý tất cả các embedding cùng một lúc**. Điều này giúp Transformer nhanh hơn nhiều, nhưng lại **làm mất đi thông tin liên quan đến thứ tự của các từ trong câu**. 
Để giải quyết vấn đề này, các tác giả của bài báo "Attention is All You Need" đã giới thiệu khái niệm [Positional Encoding](Positional%20Encoding.md) (trong bài viết này, chúng ta sẽ không đi sâu vào chi tiết). 
Có thể hiểu rằng: **mỗi từ trong câu sẽ được gán một vector đánh dấu vị trí của nó**. Vector này sẽ được cộng vào embedding của từng từ, từ đó tạo ra một vector mới dùng làm đầu vào cho mô hình. Cách làm này giúp mô hình không chỉ nhận diện nội dung của từ mà còn hiểu được vị trí của nó trong ngữ cảnh của câu, từ đó cải thiện khả năng xử lý ngữ nghĩa của toàn bộ đoạn văn.
![](https://i.imgur.com/lXa5AvH.png)

#### Cách hoạt động của self attention

Giả sử có chuỗi đầu vào gồm các vector $X=[x_1,...,x_n]$ , mỗi vector biểu diễn một token (từ hoặc từ con).
Self-attention hoạt động qua 3 bước chính:
1. **Tạo ma trận Q, K, V**
	- Mỗi token $x_i$ được nhân với 3 trọng số học được để tạo ra:
		- Query Q: đại diện cho câu hỏi, từ khóa truy vấn 
		- Key K: Đại diện cho các kết quả xuất hiện như tiêu đề,..
		- Value V: Đại diện cho nội dung bên trong 
	$Q=XW^Q$, $K=XW^K$, $V=XW^V$ 

![](https://i.imgur.com/bQ211j0.png)


2. **Tính trọng số attention**


	1. Tính điểm tương đồng (dot product) giữa Q và K:
		$$\ score_{ij}=Q_{i}\cdot K_{i}^T $$
	2. Chuẩn hóa bằng softmax để thu được hệ số attention:
		$$\alpha_{ij}=\frac{\exp(\frac{score_{ij}}{\sqrt{d_k}})}{\sum\limits_{j}\exp(\frac{score_{ij}}{\sqrt{dk}})}$$
		- Trong đó $d_k$ là số chiều của vector key (giúp ổn định gradient)
![](https://i.imgur.com/v6L2LlH.png)


3. **Tổng hợp output**
	- Mỗi token được biểu diễn mới bằng cách lấy tổng có trọng số của các V:
		$$output_i=\sum\limits_{j}\alpha_{ij}V_j$$
	- Giả sử nhập vào câu: "Hi, how are you"  và muốn transformer xuất ra "im fine"