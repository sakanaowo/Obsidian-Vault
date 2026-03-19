---
created_date: 2026-03-11
target: Min hashing
---
## Pipeline tổng quan
![[Pasted image 20260311075000.png]]

- [ ] pcy
- [ ] minhashing
- [ ] 
#  PCY

## 1. Tổng quan về bài toán

Thuật toán PCY là một phương pháp cải tiến dùng để giải quyết bài toán **tìm kiếm các cặp phổ biến (frequent pairs)** trong các tập dữ liệu lớn.

Bài toán này thường gặp khó khăn về bộ nhớ máy tính:

- **Cách tiếp cận ngây thơ (Naïve solution):** Đếm mọi tổ hợp cặp có thể có trong dữ liệu chỉ với một vòng quét. Cách này tiêu tốn bộ nhớ khổng lồ, tỉ lệ thuận với bình phương số lượng phần tử ($O(N^2)$).
- **Thuật toán A-Priori:** Cải thiện bằng cách chia làm 2 vòng (pass). Vòng 1 đếm các phần tử lẻ. Vòng 2 chỉ đếm các "cặp ứng viên" mà cả hai phần tử cấu thành đều là phần tử phổ biến. Tuy nhiên, lượng ứng viên đôi khi vẫn còn quá lớn.

Thuật toán PCY ra đời nhằm **tối ưu hóa hơn nữa** bộ nhớ bằng cách tạo ra một "bộ lọc" khắt khe hơn để giảm thiểu số lượng cặp ứng viên so với A-Priori.

---

## 2. Cách thức hoạt động của PCY

Thuật toán PCY hoạt động thông qua 2 vòng quét (pass) chính:

### Vòng 1 (Pass 1): Đếm và Băm (Hashing)

Khi đọc dữ liệu từ các giỏ hàng (baskets), thuật toán thực hiện đồng thời hai việc:

1. **Đếm phần tử lẻ:** Đếm tần suất chính xác của từng phần tử đơn lẻ (singletons) giống như thuật toán A-Priori.
2. **Tạo cặp và băm (Hashing):**
    - Phân rã mỗi giỏ hàng thành các cặp phần tử ${i, j}$. (Ví dụ: Giỏ hàng 1 chứa ${1,2,3}$ sẽ tạo ra các cặp: ${1,2}, {1,3}, {2,3}$).
    - Sử dụng một thuật toán băm (hash) để đưa các cặp này vào một trong $B$ nhóm (bucket) có giới hạn.
    - **Đếm số lượng** (count) các cặp rơi vào từng bucket thay vì đếm từng cặp cụ thể.

### Vòng 2 (Pass 2): Lọc "cặp ứng viên"

Ở vòng này, PCY xác định những cặp nào có tiềm năng trở thành "cặp phổ biến" để tiến hành đếm số lần xuất hiện thực tế. Để được chọn làm cặp ứng viên (candidate pair), một cặp ${i, j}$ phải thỏa mãn **đồng thời 2 điều kiện**:

1. **Điều kiện của A-Priori:** Cả hai phần tử đơn lẻ ${i}$ và ${j}$ đều phải là những phần tử phổ biến.
2. **Điều kiện bổ sung của PCY:** Bản thân cặp ${i,j}$ đó phải được băm vào một **bucket phổ biến** (bucket có số đếm cao ở Vòng 1).

---

## 3. Các khái niệm trọng tâm cần nhớ

- **Bucket phổ biến (Frequent Bucket):** Là bucket có tổng số lượng các cặp được băm vào nó (count of pairs) đạt mức cao (vượt qua một ngưỡng hỗ trợ nhất định).
- **Sự khác biệt cốt lõi với A-Priori:** Nằm ở việc PCY sử dụng thêm thuật toán băm vào các bucket (hashing) ở Vòng 1 làm màng lọc thứ hai, giúp giảm đáng kể số lượng cặp cần phải so sánh ở Vòng 2.
- **Xử lý đụng độ băm (Hash Collision):** Nếu một cặp ít xuất hiện ("cặp rác") vô tình bị băm trùng bucket với cặp xuất hiện nhiều ("cặp xịn"), bucket đó vẫn trở thành bucket phổ biến. Tuy nhiên, PCY không bị sai lệch kết quả vì:
    - Nếu các phần tử lẻ của "cặp rác" không phổ biến, nó bị loại ngay lập tức nhờ điều kiện 1 của Vòng 2.
    - Nếu nó lọt qua được vòng ứng viên, khi đếm tần suất thực tế, số đếm của nó vẫn sẽ rất thấp và bị loại ở bước chốt hạ.

---

## 4. Ứng dụng trong thực tế

Dù là tìm kiếm cặp hay tập hợp dữ liệu tương đồng, PCY và các thuật toán tương tự giải quyết các bài toán dữ liệu quy mô khổng lồ ($O(N)$ thay vì $O(N^2)$) trong các lĩnh vực:

- **Hệ thống gợi ý (Recommender Systems):** Gợi ý sản phẩm thường được mua cùng nhau (Market basket analysis).
- **Luật kết hợp (Association Rules):** Phân tích mối liên hệ hành vi khách hàng.
- **Phát hiện trùng lặp (Duplicate Detection):** Tìm kiếm nội dung văn bản, bài viết hoặc trang web sao chép lẫn nhau.
---

# Minhashing
## 1. Tổng quan và Mục đích

**Minhashing** là bước thứ hai cốt lõi trong quy trình xử lý và tìm kiếm các tài liệu tương đồng (similar documents).

Khi xử lý hàng triệu văn bản, việc so sánh trực tiếp từng cặp sẽ tiêu tốn một lượng thời gian khổng lồ tỷ lệ thuận với bình phương số lượng tài liệu (O(N2)). Minhashing giải quyết bài toán này bằng cách **chuyển đổi (nén) các tập hợp lớn** (các tài liệu đã được phân rã) thành các **"chữ ký" (signatures) rất ngắn gọn** (ví dụ: chỉ cần khoảng 100 số nguyên, tốn khoảng vài trăm byte), trong khi vẫn **bảo toàn được mức độ tương đồng** giữa các tài liệu gốc.

--------------------------------------------------------------------------------

## 2. Đặc tính cốt lõi (The Min-Hash Property)

Sức mạnh của Minhashing hoàn toàn dựa vào mối liên hệ toán học trực tiếp với **độ tương đồng Jaccard (Jaccard Similarity)**.

Đặc tính Min-Hash phát biểu rằng: **Xác suất để hai tài liệu (hai cột** C1​,C2​**) có cùng một giá trị băm (min-hash value) chính xác bằng với độ tương đồng Jaccard của hai tài liệu đó**. Công thức: $Pr[h(C1​)=h(C2​)]=sim(C1​,C2​)$.

Nhờ đặc tính này, sự giống nhau của hai tài liệu gốc hoàn toàn có thể được ước lượng bằng cách so sánh mức độ giống nhau của hai "chữ ký" nhỏ bé đại diện cho chúng.

---
## 3. Cách thức hoạt động

Quá trình nén tài liệu thành chữ ký thông qua Minhashing diễn ra qua các bước sau:

A. Biểu diễn dữ liệu bằng Ma trận nhị phân (Boolean Matrix)

- **Hàng (Rows):** Đại diện cho các phần tử duy nhất trong toàn bộ dữ liệu (các shingle).
- **Cột (Columns):** Đại diện cho các tài liệu (documents).
- **Giá trị:** Ô ở hàng e và cột s chứa giá trị **1** nếu tài liệu s chứa phần tử e, ngược lại là 0. Các ma trận này thường vô cùng lớn và cực kỳ thưa thớt (rất nhiều số 0).

B. Nguyên lý lý thuyết: Hoán vị ngẫu nhiên (Random Permutation)

Để tính giá trị Min-Hash cho một cột (tài liệu):

1. Tiến hành **xáo trộn (hoán vị) ngẫu nhiên** thứ tự tất cả các hàng của ma trận.
2. Duyệt dọc theo cột đó từ trên xuống dưới. **Chỉ số của hàng đầu tiên có chứa giá trị 1** chính là giá trị Min-Hash của tài liệu.
3. Lặp lại quá trình này bằng cách chọn K phép hoán vị ngẫu nhiên độc lập (ví dụ K=100) để tạo ra một cột "chữ ký" gồm K số nguyên.

C. Triển khai thực tế: Thuật toán quét một lần (Implementation Trick)

Việc xáo trộn vật lý hàng triệu hàng dữ liệu là bất khả thi. Thực tế, người ta dùng K hàm băm (hash functions) ngẫu nhiên k1​,k2​,...,kK​ để mô phỏng thứ tự hoán vị chỉ với **một vòng quét dữ liệu duy nhất (one-pass)**:

1. **Khởi tạo:** Thiết lập mảng chữ ký cho cột C, sig(C)[i]=∞ (vô cực) cho mọi hàm băm i.
2. **Quét hàng:** Quét lần lượt các hàng. Giả sử đang ở hàng j, nếu cột C có giá trị **1**.
3. **Cập nhật:** Tính các giá trị băm ki​(j). Nếu ki​(j)<sig(C)[i], ta cập nhật giá trị mới: sig(C)[i]←ki​(j). Sau khi duyệt hết, ta thu được mảng chữ ký gọn nhẹ chứa các giá trị nhỏ nhất từ các hàm băm.

--------------------------------------------------------------------------------

## 4. Vị trí trong chuỗi xử lý (The Big Picture)

Minhashing không đứng một mình mà nằm ở bước giữa trong hệ thống 3 bước tìm kiếm tài liệu tương đồng:

1. **Shingling (Cắt văn bản):** Chuyển đổi tài liệu văn bản thô thành các tập hợp các chuỗi con (k-shingles).
2. **Minhashing (Nén thành chữ ký):** Nén các ma trận tập hợp khổng lồ thành các chữ ký nhỏ gọn (signatures) nhưng vẫn bảo toàn độ tương đồng Jaccard.
3. **LSH - Locality-Sensitive Hashing (Lọc ứng viên):** Lấy các chữ ký do Minhashing tạo ra, chia thành các dải (bands) và băm vào các bucket để nhanh chóng tìm ra các **"cặp ứng viên" (candidate pairs)** có khả năng cao là giống nhau, loại bỏ phần lớn các cặp rác mà không cần so sánh toàn bộ cơ sở dữ liệu