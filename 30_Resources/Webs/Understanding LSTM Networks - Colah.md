---
title: "Understanding LSTM Networks"
author: Christopher Olah
source: "https://colah.github.io/posts/2015-08-Understanding-LSTMs/"
date_published: 2015-08-27
date_translated: 2026-04-19
type: web-article
tags:
  - LSTM
  - RNN
  - deep-learning
  - source-note
aliases:
  - "Colah LSTM"
  - "Understanding LSTMs"
related:
  - "[[LSTM]]"
  - "[[Recurrent Neural Network]]"
  - "[[GRU]]"
  - "[[Backpropagation Through Time]]"
---

# Hiểu về mạng LSTM (Understanding LSTM Networks)

> **Nguồn gốc:** Bài viết kinh điển của **Christopher Olah** (2015), được coi là tài liệu trực quan nhất về LSTM trên internet. Bài viết này dùng các sơ đồ tuyệt đẹp để giải thích từng bước cách LSTM hoạt động — từ vấn đề của RNN thường, đến ý tưởng cốt lõi, đến từng cổng cụ thể.

---

## 1. Recurrent Neural Networks (Mạng nơ-ron hồi tiếp)

Con người không bắt đầu suy nghĩ từ con số 0 mỗi giây. Khi bạn đọc bài viết này, bạn hiểu mỗi từ dựa trên sự hiểu biết về các từ trước đó. Bạn không vứt bỏ mọi thứ và bắt đầu nghĩ lại từ đầu. Suy nghĩ của bạn có **tính liên tục** (persistence).

Mạng nơ-ron truyền thống (feedforward) không thể làm điều này, và đây là một thiếu sót lớn. Ví dụ, hãy tưởng tượng bạn muốn phân loại loại sự kiện đang xảy ra tại mỗi thời điểm trong một bộ phim. Không rõ mạng nơ-ron truyền thống có thể dùng suy luận về các sự kiện trước đó trong phim để suy ra các sự kiện sau như thế nào.

**Mạng nơ-ron hồi tiếp** (Recurrent Neural Networks — RNN) giải quyết vấn đề này. Chúng là mạng có **vòng lặp** (loops) bên trong, cho phép thông tin được duy trì qua thời gian.

![[assets/attachments/colah-understanding-lstms/RNN-rolled.png]]
_Mạng nơ-ron hồi tiếp có vòng lặp._

Trong sơ đồ trên, một khối mạng nơ-ron $A$ nhận đầu vào $x_t$ và xuất ra giá trị $h_t$. Vòng lặp cho phép thông tin được truyền từ bước này sang bước tiếp theo của mạng.

Các vòng lặp này khiến RNN trông có vẻ bí ẩn. Nhưng nếu suy nghĩ kỹ hơn, chúng thực ra không khác mạng nơ-ron bình thường nhiều lắm. Một RNN có thể được xem như **nhiều bản sao** của cùng một mạng, mỗi bản truyền một thông điệp (message) cho bản kế tiếp. Hãy xem điều gì xảy ra khi ta **trải ra** (unroll) vòng lặp:

![[assets/attachments/colah-understanding-lstms/RNN-unrolled.png]]
_Một RNN đã được trải ra (unrolled)._

Cấu trúc dạng chuỗi (chain) này cho thấy RNN có **quan hệ mật thiết với chuỗi và danh sách** (sequences and lists). Chúng là kiến trúc mạng nơ-ron tự nhiên nhất để xử lý loại dữ liệu này.

Và RNN chắc chắn đã được sử dụng rộng rãi! Trong vài năm qua, đã có những thành công đáng kinh ngạc khi áp dụng RNN vào đủ loại bài toán: nhận dạng giọng nói, mô hình ngôn ngữ, dịch máy, mô tả ảnh (image captioning)... Danh sách còn dài. Để biết thêm về những kỳ tích có thể đạt được với RNN, xem bài blog tuyệt vời của Andrej Karpathy: [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

Chìa khóa cho những thành công này chính là việc sử dụng **"LSTMs"** — một loại RNN rất đặc biệt, hoạt động tốt hơn phiên bản tiêu chuẩn rất, rất nhiều cho hầu hết mọi bài toán. Hầu như tất cả các kết quả thú vị dựa trên RNN đều đạt được nhờ LSTM. Bài viết này sẽ khám phá LSTM.

---

## 2. Vấn đề phụ thuộc dài hạn (The Problem of Long-Term Dependencies)

Một trong những điểm hấp dẫn của RNN là ý tưởng rằng chúng có thể **kết nối thông tin trước đó với tác vụ hiện tại** — ví dụ, dùng các khung hình video trước để hiểu khung hình hiện tại. Nếu RNN thật sự làm được điều này thì chúng sẽ cực kỳ hữu ích. Nhưng liệu chúng có thể? Câu trả lời là: **tùy trường hợp**.

### 2.1. Khi khoảng cách ngắn: RNN làm tốt

Đôi khi ta chỉ cần nhìn thông tin **gần đây** để thực hiện tác vụ hiện tại. Ví dụ, xét một mô hình ngôn ngữ cố gắng dự đoán từ tiếp theo dựa trên các từ trước đó. Nếu ta cố dự đoán từ cuối trong câu _"the clouds are in the **sky**"_, ta không cần thêm ngữ cảnh nào — rõ ràng từ tiếp theo sẽ là "sky". Trong trường hợp này, khoảng cách (gap) giữa thông tin liên quan và vị trí cần dùng là **nhỏ**, RNN có thể học cách sử dụng thông tin quá khứ.

![[assets/attachments/colah-understanding-lstms/RNN-shorttermdepdencies.png]]
_Khi khoảng cách ngữ cảnh ngắn, RNN hoạt động tốt._

### 2.2. Khi khoảng cách dài: RNN thất bại

Nhưng cũng có trường hợp ta cần **nhiều ngữ cảnh hơn**. Xét việc dự đoán từ cuối trong đoạn: _"I grew up in **France**... I speak fluent **French**."_ Thông tin gần đây gợi ý từ tiếp theo có thể là tên một ngôn ngữ, nhưng muốn thu hẹp xuống _ngôn ngữ nào_, ta cần ngữ cảnh "France" từ **rất xa** phía trước. Hoàn toàn có thể xảy ra trường hợp khoảng cách giữa thông tin liên quan và điểm cần dùng trở nên **rất lớn**.

Thật không may, khi khoảng cách đó tăng, RNN trở nên **không thể học** cách kết nối thông tin.

![[assets/attachments/colah-understanding-lstms/RNN-longtermdependencies.png]]
_Mạng nơ-ron gặp khó khăn với phụ thuộc dài hạn._

> [!NOTE] Tại sao RNN thất bại ở khoảng cách dài?
> Về lý thuyết, RNN hoàn toàn có khả năng xử lý các **phụ thuộc dài hạn** (long-term dependencies). Một người có thể cẩn thận chọn tham số cho RNN để giải các bài toán đồ chơi (toy problems) dạng này. Nhưng buồn thay, trên thực tế, RNN dường như không thể **học** được chúng. Vấn đề này đã được nghiên cứu sâu bởi [Hochreiter (1991)](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf) và [Bengio và cộng sự (1994)](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf), những người đã tìm ra các lý do nền tảng giải thích tại sao điều này khó khăn.
>
> **Cơ chế:** Khi lan truyền ngược qua thời gian, gradient phải nhân liên tiếp với ma trận trọng số $W_{hh}$. Nếu eigenvalue lớn nhất của $W_{hh}$ nhỏ hơn 1, gradient **biến mất** (vanishing). Nếu lớn hơn 1, gradient **bùng nổ** (exploding). Xem chi tiết trong [[Backpropagation Through Time]].

May mắn thay, LSTM không gặp vấn đề này!

---

## 3. Mạng LSTM (LSTM Networks)

> [!NOTE] ELI5
> Hãy tưởng tượng bạn đang đọc một cuốn sách dài. RNN thường giống như một người **chỉ nhớ vài trang gần đây nhất** — đọc đến chương 10 thì quên mất nội dung chương 1. LSTM giống như người có thêm **một cuốn sổ tay** (cell state) — họ ghi lại những thông tin quan trọng, gạch bỏ những gì không cần nữa, và chỉ cần liếc sổ tay là nhớ lại ngay. Ba cây bút màu khác nhau (ba cổng) giúp họ quyết định: xóa gì, ghi thêm gì, và đọc lại phần nào.

**Long Short Term Memory** (Bộ nhớ ngắn-dài hạn) — thường gọi tắt là **"LSTM"** — là một loại RNN đặc biệt, có khả năng **học các phụ thuộc dài hạn**. Chúng được giới thiệu bởi [Hochreiter & Schmidhuber (1997)](http://www.bioinf.jku.at/publications/older/2604.pdf), và sau đó được cải tiến và phổ biến bởi nhiều người trong các công trình tiếp theo[^1]. LSTM hoạt động cực kỳ tốt trên rất nhiều loại bài toán, và hiện được sử dụng rộng rãi.

LSTM được thiết kế **rõ ràng** để tránh vấn đề phụ thuộc dài hạn. **Ghi nhớ thông tin trong thời gian dài thực chất là hành vi mặc định của chúng**, chứ không phải thứ chúng phải vật lộn để học!

### 3.1. Cấu trúc lặp lại của RNN chuẩn vs LSTM

Mọi RNN đều có dạng **một chuỗi các module lặp lại** (repeating modules) của mạng nơ-ron. Trong RNN chuẩn, module lặp lại này có cấu trúc rất đơn giản — ví dụ chỉ một lớp **tanh** duy nhất.

![[assets/attachments/colah-understanding-lstms/LSTM3-SimpleRNN.png]]
_Module lặp lại trong RNN chuẩn chứa một lớp duy nhất (tanh)._

LSTM cũng có cấu trúc dạng chuỗi tương tự, nhưng module lặp lại có cấu trúc **khác hẳn**. Thay vì chỉ có một lớp mạng nơ-ron, có **bốn lớp**, tương tác với nhau theo cách rất đặc biệt.

![[assets/attachments/colah-understanding-lstms/LSTM3-chain.png]]
_Module lặp lại trong LSTM chứa bốn lớp tương tác._

Đừng lo lắng về chi tiết ngay bây giờ. Ta sẽ đi qua sơ đồ LSTM **từng bước** ở phần sau. Trước tiên, hãy làm quen với **ký hiệu** mà ta sẽ sử dụng.

![[assets/attachments/colah-understanding-lstms/LSTM2-notation.png]]
_Chú giải ký hiệu trong sơ đồ LSTM._

Trong sơ đồ trên:

- Mỗi **đường** mang theo **toàn bộ một vector**, từ đầu ra của nút này đến đầu vào của các nút khác.
- Các **hình tròn màu hồng** biểu thị phép toán **theo phần tử** (pointwise operations), ví dụ cộng vector hoặc nhân Hadamard.
- Các **hộp màu vàng** là các **lớp mạng nơ-ron đã học** (learned neural network layers).
- Đường **gộp** (merge) biểu thị phép **nối** (concatenation), đường **tách** (fork) biểu thị nội dung được **sao chép** và gửi đến nhiều vị trí khác nhau.

---

## 4. Ý tưởng cốt lõi đằng sau LSTM (The Core Idea Behind LSTMs)

Chìa khóa của LSTM là **trạng thái ô** (cell state) — đường ngang chạy qua đỉnh sơ đồ.

Cell state giống như một **băng chuyền** (conveyor belt). Nó chạy thẳng xuống toàn bộ chuỗi, chỉ với một vài tương tác tuyến tính nhỏ. Thông tin rất dễ dàng chảy dọc theo nó mà **không bị thay đổi**.

![[assets/attachments/colah-understanding-lstms/LSTM3-C-line.png]]
_Trạng thái ô (cell state) — đường ngang phía trên chạy xuyên suốt chuỗi._

> [!NOTE] Tại sao cell state quan trọng?
> Đây chính là cơ chế giải quyết **vanishing gradient**. Trong RNN thường, thông tin phải đi qua nhiều phép nhân ma trận liên tiếp → gradient bị thu nhỏ dần. Trong LSTM, cell state được cập nhật bằng **phép cộng** ($C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$), cho phép gradient chảy gần như nguyên vẹn qua nhiều bước thời gian.

LSTM có khả năng **loại bỏ** hoặc **thêm** thông tin vào cell state, được điều chỉnh cẩn thận bởi các cấu trúc gọi là **cổng** (gates).

Cổng là cách để **tùy chọn** cho thông tin đi qua. Chúng bao gồm một lớp mạng nơ-ron **sigmoid** và một phép **nhân theo phần tử** (pointwise multiplication).

![[assets/attachments/colah-understanding-lstms/LSTM3-gate.png]]
_Cấu trúc một cổng: sigmoid + pointwise multiplication._

Lớp sigmoid xuất ra các số trong khoảng $[0, 1]$, mô tả **bao nhiêu phần** của mỗi thành phần nên được cho qua:

- Giá trị **0** nghĩa là _"không cho gì qua cả"_
- Giá trị **1** nghĩa là _"cho mọi thứ đi qua!"_

Một LSTM có **ba** cổng như vậy, để bảo vệ và kiểm soát cell state.

---

## 5. Đi qua LSTM từng bước (Step-by-Step LSTM Walk Through)

### 5.1. Bước 1 — Cổng quên (Forget Gate): Quyết định bỏ gì

Bước đầu tiên trong LSTM là quyết định **thông tin nào sẽ bị loại bỏ** khỏi cell state. Quyết định này được đưa ra bởi một lớp sigmoid gọi là **"forget gate layer"** (lớp cổng quên). Nó nhìn vào $h_{t-1}$ (hidden state trước) và $x_t$ (input hiện tại), và xuất ra một số trong khoảng $[0, 1]$ cho **mỗi phần tử** trong cell state $C_{t-1}$.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

- Giá trị **1** nghĩa là _"giữ nguyên hoàn toàn"_
- Giá trị **0** nghĩa là _"loại bỏ hoàn toàn"_

![[assets/attachments/colah-understanding-lstms/LSTM3-focus-f.png]]
_Cổng quên (forget gate) — quyết định loại bỏ thông tin nào khỏi cell state._

> **Ví dụ thực tế:** Trong bài toán mô hình ngôn ngữ, cell state có thể chứa **giới tính** của chủ ngữ hiện tại để dùng đúng đại từ nhân xưng. Khi gặp **chủ ngữ mới**, ta muốn **quên** giới tính của chủ ngữ cũ.

### 5.2. Bước 2 — Cổng đầu vào (Input Gate): Quyết định lưu gì mới

Bước tiếp theo là quyết định **thông tin mới nào sẽ được lưu** vào cell state. Bước này gồm hai phần:

1. **Lớp sigmoid** gọi là **"input gate layer"** — quyết định **giá trị nào** ta sẽ cập nhật.
2. **Lớp tanh** — tạo ra vector các **giá trị ứng viên** (candidate values) $\tilde{C}_t$ có thể được thêm vào state.

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

Ở bước tiếp theo, ta sẽ **kết hợp** hai thứ này để tạo ra bản cập nhật cho state.

![[assets/attachments/colah-understanding-lstms/LSTM3-focus-i.png]]
_Cổng đầu vào (input gate) — quyết định thông tin mới nào sẽ được lưu._

> **Ví dụ thực tế:** Trong mô hình ngôn ngữ, ta muốn **thêm giới tính** của chủ ngữ mới vào cell state, thay thế giới tính cũ mà ta đang quên.

### 5.3. Bước 3 — Cập nhật Cell State

Bây giờ là lúc cập nhật cell state cũ $C_{t-1}$ thành cell state mới $C_t$. Các bước trước đã quyết định **làm gì**, giờ ta chỉ cần **thực hiện**.

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

Ta nhân state cũ với $f_t$ — **quên** những gì ta đã quyết định quên. Rồi cộng thêm $i_t \odot \tilde{C}_t$ — **thông tin ứng viên mới**, được scale bởi mức độ ta quyết định cập nhật cho mỗi giá trị.

![[assets/attachments/colah-understanding-lstms/LSTM3-focus-C.png]]
_Cập nhật cell state: quên cái cũ + thêm cái mới._

> **Ví dụ thực tế:** Trong mô hình ngôn ngữ, đây là nơi ta thực sự **xóa thông tin** về giới tính chủ ngữ cũ và **thêm thông tin mới**, đúng như đã quyết định ở các bước trước.

> [!NOTE] Tại sao phép cộng quan trọng?
> Công thức $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ dùng **phép cộng** chứ không phải phép nhân ma trận. Khi tính gradient ngược:
> $$\frac{\partial C_t}{\partial C_{t-1}} = \text{diag}(f_t)$$
> Đây là **ma trận đường chéo** với các phần tử trong $[0, 1]$, không phải tích ma trận đầy đủ. Nếu cổng quên mở ($f_t \approx 1$), gradient chảy gần như **nguyên vẹn** — đây chính là cơ chế giải quyết vanishing gradient!

### 5.4. Bước 4 — Cổng đầu ra (Output Gate): Quyết định xuất gì

Cuối cùng, ta cần quyết định **xuất ra cái gì**. Đầu ra sẽ dựa trên cell state, nhưng là phiên bản **được lọc**. Quy trình:

1. Chạy **lớp sigmoid** để quyết định **phần nào** của cell state ta sẽ xuất ra.
2. Đưa cell state qua **tanh** (để giá trị nằm trong $[-1, 1]$).
3. **Nhân** kết quả tanh với đầu ra của sigmoid — chỉ xuất những phần ta đã quyết định.

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

![[assets/attachments/colah-understanding-lstms/LSTM3-focus-o.png]]
_Cổng đầu ra (output gate) — quyết định xuất phần nào của cell state._

> **Ví dụ thực tế:** Trong mô hình ngôn ngữ, vì vừa thấy một chủ ngữ, mạng có thể muốn xuất thông tin **liên quan đến động từ** — ví dụ, chủ ngữ là số ít hay số nhiều, để biết phải chia động từ thế nào nếu động từ xuất hiện tiếp theo.

---

## 6. Các biến thể của LSTM (Variants on Long Short Term Memory)

Những gì mô tả ở trên là LSTM khá chuẩn. Nhưng không phải mọi LSTM đều giống nhau. Thực tế, hầu như mọi bài báo liên quan đến LSTM đều dùng một phiên bản hơi **khác biệt**. Các khác biệt thường nhỏ, nhưng đáng để đề cập.

### 6.1. Peephole Connections (Kết nối nhìn trộm)

Một biến thể LSTM phổ biến, được giới thiệu bởi [Gers & Schmidhuber (2000)](ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf), là thêm **"peephole connections"** — cho phép các lớp cổng **nhìn vào cell state** $C_{t-1}$.

![[assets/attachments/colah-understanding-lstms/LSTM3-var-peepholes.png]]
_Peephole connections — các cổng có thể nhìn trực tiếp vào cell state._

Sơ đồ trên thêm peephole cho **tất cả** các cổng, nhưng nhiều bài báo chỉ thêm cho **một số** cổng.

> **Ý nghĩa:** Trong LSTM chuẩn, các cổng chỉ nhìn $h_{t-1}$ và $x_t$. Với peephole, chúng còn nhìn được $C_{t-1}$ (hoặc $C_t$), tức là biết thêm **nội dung thực sự** đang được lưu trữ. Điều này hữu ích cho các bài toán cần **biết chính xác trạng thái hiện tại** để ra quyết định cổng.

### 6.2. Coupled Forget and Input Gates (Cổng quên và cổng đầu vào ghép đôi)

Một biến thể khác là dùng cổng quên và cổng đầu vào **ghép đôi**. Thay vì quyết định riêng biệt _quên gì_ và _thêm gì mới_, ta đưa ra **cả hai quyết định cùng lúc**. Ta chỉ quên khi sắp thêm thứ gì mới vào vị trí đó. Ta chỉ thêm giá trị mới khi quên thứ gì đó cũ hơn.

$$C_t = (1 - i_t) \odot C_{t-1} + i_t \odot \tilde{C}_t$$

![[assets/attachments/colah-understanding-lstms/LSTM3-var-tied.png]]
_Coupled gates — cổng quên = 1 − cổng đầu vào._

> **Ý nghĩa:** Ràng buộc $f_t = 1 - i_t$ đảm bảo **tổng trọng số luôn bằng 1** — giống phép nội suy (interpolation) giữa giá trị cũ và giá trị mới. Giảm số tham số và đôi khi ổn định hơn.

### 6.3. GRU — Gated Recurrent Unit (Đơn vị hồi tiếp có cổng)

Biến thể đáng chú ý hơn là **Gated Recurrent Unit** (GRU), được giới thiệu bởi [Cho và cộng sự (2014)](http://arxiv.org/pdf/1406.1078v3.pdf). GRU **gộp** cổng quên và cổng đầu vào thành một **"update gate"** (cổng cập nhật) duy nhất. Nó cũng **gộp** cell state và hidden state, cùng một số thay đổi khác. Mô hình kết quả **đơn giản hơn** LSTM chuẩn, và ngày càng phổ biến.

![[assets/attachments/colah-understanding-lstms/LSTM3-var-GRU.png]]
_Gated Recurrent Unit (GRU) — đơn giản hơn LSTM._

> **So sánh GRU vs LSTM:**
>
> | Tiêu chí   | LSTM                                     | GRU                                              |
> | ---------- | ---------------------------------------- | ------------------------------------------------ |
> | Cổng       | 3 (quên, đầu vào, đầu ra)                | 2 (reset, update)                                |
> | Trạng thái | 2 (cell state $C_t$, hidden state $H_t$) | 1 (hidden state $H_t$ duy nhất)                  |
> | Tham số    | Nhiều hơn (~gấp 1.33 lần)                | Ít hơn                                           |
> | Hiệu năng  | Thường tốt hơn ở chuỗi rất dài           | Tương đương hoặc tốt hơn ở chuỗi ngắn-trung bình |

Đây chỉ là một vài biến thể LSTM nổi bật nhất. Còn nhiều biến thể khác, như **Depth Gated RNNs** của [Yao và cộng sự (2015)](http://arxiv.org/pdf/1508.03790v2.pdf). Cũng có những cách tiếp cận hoàn toàn khác để giải quyết phụ thuộc dài hạn, như **Clockwork RNNs** của [Koutnik và cộng sự (2014)](http://arxiv.org/pdf/1402.3511v1.pdf).

### 6.4. Biến thể nào tốt nhất?

[Greff và cộng sự (2015)](http://arxiv.org/pdf/1503.04069.pdf) đã so sánh kỹ các biến thể phổ biến, kết luận rằng chúng **gần như tương đương nhau**. [Jozefowicz và cộng sự (2015)](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf) thử nghiệm hơn **10.000 kiến trúc RNN**, tìm ra một số kiến trúc hoạt động tốt hơn LSTM trên một số tác vụ nhất định.

---

## 7. Kết luận (Conclusion)

Trước đó, tôi đã nhắc đến các kết quả đáng chú ý mà mọi người đạt được với RNN. Gần như **tất cả** đều đạt được nhờ LSTM. LSTM thực sự hoạt động tốt hơn nhiều cho hầu hết mọi tác vụ!

Khi viết dưới dạng tập hợp các phương trình, LSTM trông khá đáng sợ. Hy vọng rằng, việc đi qua chúng **từng bước** trong bài viết này đã khiến chúng trở nên **dễ tiếp cận** hơn.

LSTM là một bước tiến lớn trong những gì ta có thể đạt được với RNN. Tự nhiên ta sẽ hỏi: _liệu có bước tiến lớn tiếp theo?_ Một ý kiến phổ biến trong giới nghiên cứu là: **"Có! Và đó là Attention!"** Ý tưởng là cho phép mỗi bước của RNN **chọn** thông tin cần nhìn từ một bộ sưu tập lớn hơn. Ví dụ, nếu bạn dùng RNN để tạo mô tả cho một bức ảnh, nó có thể chọn **một phần** của ảnh để nhìn cho mỗi từ nó xuất ra. Thực tế, [Xu và cộng sự (2015)](http://arxiv.org/pdf/1502.03044v2.pdf) làm đúng điều này — đây có thể là điểm khởi đầu thú vị nếu bạn muốn khám phá attention!

> [!NOTE] Bối cảnh lịch sử (2015 → 2026)
> Bài viết này được viết năm **2015**, trước khi **Transformer** (2017) ra đời. Attention mà Colah nhắc đến ở đây là **attention trong RNN** (Bahdanau attention, 2014). Kể từ đó, **self-attention** trong Transformer đã thay thế RNN/LSTM trong hầu hết các tác vụ NLP. Tuy nhiên, hiểu LSTM vẫn quan trọng vì:
>
> 1. Nó là nền tảng để hiểu **tại sao attention ra đời**
> 2. LSTM vẫn được dùng trong nhiều ứng dụng **time series**, **audio processing**, và các tác vụ mà dữ liệu thực sự có tính tuần tự mạnh
> 3. Các khái niệm gating, cell state đã ảnh hưởng đến nhiều kiến trúc hiện đại

---

## Lời cảm ơn (Acknowledgments)

Tôi biết ơn nhiều người đã giúp tôi hiểu rõ hơn về LSTM, nhận xét về các hình minh họa, và cung cấp phản hồi cho bài viết này.

Tôi rất biết ơn các đồng nghiệp tại Google vì phản hồi hữu ích, đặc biệt là [Oriol Vinyals](http://research.google.com/pubs/OriolVinyals.html), [Greg Corrado](http://research.google.com/pubs/GregCorrado.html), [Jon Shlens](http://research.google.com/pubs/JonathonShlens.html), [Luke Vilnis](http://people.cs.umass.edu/~luke/), và [Ilya Sutskever](http://www.cs.toronto.edu/~ilya/). Tôi cũng cảm ơn nhiều bạn bè và đồng nghiệp khác, bao gồm [Dario Amodei](https://www.linkedin.com/pub/dario-amodei/4/493/393) và [Jacob Steinhardt](http://cs.stanford.edu/~jsteinhardt/). Đặc biệt cảm ơn [Kyunghyun Cho](http://www.kyunghyuncho.me/) vì những trao đổi rất chu đáo về các sơ đồ của tôi.

[^1]: Ngoài các tác giả gốc, nhiều người đã đóng góp cho LSTM hiện đại. Danh sách không đầy đủ bao gồm: Felix Gers, Fred Cummins, Santiago Fernandez, Justin Bayer, Daan Wierstra, Julian Togelius, Faustino Gomez, Matteo Gagliolo, và [Alex Graves](https://scholar.google.com/citations?user=DaFHynwAAAAJ&hl=en).

---

## Liên kết nội bộ

- **Concept note:** [[LSTM]] — ghi chú tổng hợp về kiến trúc LSTM
- **Concept note:** [[GRU]] — biến thể đơn giản hơn của LSTM
- **Concept note:** [[Recurrent Neural Network]] — kiến trúc nền tảng
- **Concept note:** [[Backpropagation Through Time]] — cơ chế huấn luyện RNN
- **Buổi học liên quan:** [[Buổi 42 - Tuần 12]] — BPTT chi tiết
- **Buổi học liên quan:** [[Buổi 43 - Tuần 12]] — LSTM implementation
