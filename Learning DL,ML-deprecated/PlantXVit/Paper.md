### **Tóm tắt (Abstract)**  
**Nguyên văn:**  
Plant diseases are the primary cause of crop losses globally... In this study, a Vision Transformer enabled Convolutional Neural Network model called "PlantXViT" is proposed for plant disease identification... The proposed PlantXViT network performs better than five state-of-the-art methods on all five datasets...

**Dịch sang tiếng Việt:**  
Bệnh cây trồng là nguyên nhân chính gây thiệt hại mùa vụ trên toàn cầu, ảnh hưởng lớn đến kinh tế thế giới. Để giải quyết vấn đề này, các giải pháp nông nghiệp thông minh kết hợp Internet vạn vật (IoT) và máy học (machine learning) đã được phát triển để phát hiện và kiểm soát bệnh sớm. Trong nghiên cứu này, một mô hình mạng nơ-ron tích hợp (CNN) kết hợp với Vision Transformer, được gọi là "PlantXViT", được đề xuất để nhận diện bệnh cây trồng. Mô hình này kết hợp khả năng của CNN truyền thống và Vision Transformer để nhận diện hiệu quả nhiều loại bệnh trên các loại cây trồng khác nhau. PlantXViT có cấu trúc nhẹ với chỉ 0,8 triệu tham số có thể huấn luyện, phù hợp cho các dịch vụ nông nghiệp thông minh dựa trên IoT. Hiệu suất của PlantXViT được đánh giá trên năm tập dữ liệu công khai, vượt trội so với năm phương pháp tiên tiến khác, đạt độ chính xác trung bình trên 93,55% (Apple), 92,59% (Maize) và 98,33% (Rice), ngay cả trong điều kiện nền phức tạp. Độ giải thích của mô hình được đánh giá bằng bản đồ kích hoạt lớp có trọng số gradient (Grad-CAM) và giải thích mô hình không phụ thuộc (LIME).

**Từ khóa:** Nhận diện bệnh cây trồng, Vision Transformer, Mạng nơ-ron tích hợp, Học sâu, Grad-CAM, LIME.

---

### **Giới thiệu (Introduction)**  
**Nguyên văn:**  
Human population will surpass 10 billion in the next 30 years... Plant diseases alone are responsible for 20-40% of crop yield losses... Several ML approaches have been suggested... In recent years, the focus is shifted towards deep learning (DL) algorithms...

**Dịch sang tiếng Việt:**  
Dân số thế giới dự kiến vượt 10 tỷ người trong 30 năm tới, dẫn đến nhu cầu lương thực tăng cao. Để đảm bảo nông nghiệp bền vững, việc ngăn chặn sâu bệnh và bệnh cây trồng là tối quan trọng, vì bệnh cây trồng gây thiệt hại 20-40% sản lượng mùa vụ, ảnh hưởng lớn đến ngành nông nghiệp. Các giải pháp nông nghiệp thông minh kết hợp IoT và máy học (ML) đã được nghiên cứu để phát hiện và kiểm soát bệnh sớm. Nhiều phương pháp ML như máy vector hỗ trợ (SVM), mạng nơ-ron nhân tạo (ANN), Naive Bayes, và phân cụm k-means đã được đề xuất. Gần đây, sự tập trung chuyển sang các thuật toán học sâu (DL) nhờ vào lượng dữ liệu lớn, sức mạnh tính toán và các phương pháp huấn luyện hiệu quả. Đặc biệt, các kiến trúc mạng nơ-ron tích hợp (CNN) như AlexNet, GoogleNet, VGG16, và ResNet đã mang lại kết quả nổi bật trong việc phát hiện bệnh cây trồng. Ngoài ra, xu hướng sử dụng Vision Transformer (ViT) trong học sâu dựa trên thị giác cũng đang nổi lên, nhưng chưa được khám phá nhiều trong ứng dụng bệnh lý cây trồng. Nghiên cứu này đề xuất một mô hình nhẹ, kết hợp CNN và ViT, mang tên PlantXViT, để nhận diện bệnh cây trồng với độ chính xác và khả năng giải thích cao.

---

### **Công trình liên quan (Related Works)**  
**Nguyên văn:**  
With the impressive performance of CNN in computer vision... The initial work on plant disease detection using CNN was carried out by Mohanty et al... A review of existing works indicates that CNN models with attention mechanisms have demonstrated higher accuracy...

**Dịch sang tiếng Việt:**  
Với hiệu suất ấn tượng của CNN trong lĩnh vực thị giác máy tính, các nhà nghiên cứu ngày càng quan tâm đến việc phát triển các mô hình học sâu cho nhận diện bệnh cây trồng tự động. Công trình đầu tiên sử dụng CNN được thực hiện bởi Mohanty et al. [11], đạt độ chính xác 99,35% trên tập dữ liệu PlantVillage với 54.305 ảnh thuộc 38 lớp. Barbedo [12] phân tích ảnh hưởng của kích thước và sự đa dạng tập dữ liệu, đạt độ chính xác 87% trên 1.383 ảnh thuộc 56 lớp. Too et al. [24] so sánh các kiến trúc VGG16, Inception v4, ResNet và DenseNet, trong đó DenseNet121 đạt độ chính xác cao nhất (99,75%). Chen et al. [25, 16, 18, 19, 28] phát triển các mô hình kết hợp cơ chế chú ý (attention mechanism) và đạt độ chính xác cao trên các tập dữ liệu như PlantVillage, Maize, và Rice. Zhao et al. [20] sử dụng mô-đun chú ý cải tiến và đạt 99,55% trên các tập dữ liệu ngô, cà chua và khoai tây. Lu et al. [29] kết hợp GhostNet và ViT, đạt 98,14% trên tập dữ liệu GLDP12k. Tuy nhiên, các mô hình này chưa được phân tích kỹ về khả năng giải thích (interpretability). Nghiên cứu này đề xuất PlantXViT, một mô hình kết hợp CNN và ViT, cải thiện hiệu suất phân loại và khả năng giải thích trên nhiều loại bệnh cây trồng.

---

### **Mô hình đề xuất: PlantXViT (Explainable Vision Transformer Enabled Convolutional Neural Network)**  
**Nguyên văn:**  
This section is devoted to the proposed model PlantXViT that uses ViT for plant disease detection and identification... The model consists of two Conv blocks of VGG16 network pre-trained on the Imagenet dataset...

**Dịch sang tiếng Việt:**  
Phần này trình bày mô hình PlantXViT, sử dụng Vision Transformer (ViT) để phát hiện và nhận diện bệnh cây trồng. PlantXViT kết hợp CNN và ViT, tận dụng khả năng trích xuất đặc trưng cục bộ của CNN và đặc trưng toàn cục của ViT. Mô hình bao gồm:  
1. **Khối CNN**: Sử dụng hai khối tích chập (Conv) của VGG16 được huấn luyện trước trên ImageNet, mỗi khối gồm hai lớp tích chập và một lớp gộp cực đại (max pooling), tạo ra đầu ra kích thước 56x56x128.  
2. **Khối Inception v7**: Tăng cường khả năng học đặc trưng đa tỷ lệ, tạo đầu ra 56x56x512.  
3. **Khối Transformer**: Các đặc trưng được chia thành các mảng (patch) 5x5, sau đó được chiếu tuyến tính và xử lý bởi bốn khối mã hóa transformer với cơ chế tự chú ý đa đầu (MHA) và mạng perceptron đa lớp (MLP).  
4. **Lớp đầu ra**: Một lớp gộp trung bình toàn cục (global average pooling) và một lớp kết nối đầy đủ (fully connected) với kích hoạt softmax, số nơ-ron bằng số lớp trong tập dữ liệu.  

Mô hình có tổng cộng 850.500 tham số có thể huấn luyện, phù hợp với các thiết bị IoT. PlantXViT được huấn luyện và đánh giá trên năm tập dữ liệu công khai, với kết quả được trình bày ở phần tiếp theo.

---

### **Kết quả và Thảo luận (Results and Discussion)**  
**Nguyên văn:**  
The plant disease detection model PlantXViT was developed using five publicly available datasets... PlantXViT achieves $93.55 \%, 89.24 \%, 92.59 \%, 98.86 \%$, and $98.33 \%$ overall accuracy on Apple, Embrapa, Maize, PlantVillage, and Rice datasets, respectively...

**Dịch sang tiếng Việt:**  
Mô hình PlantXViT được phát triển và đánh giá trên năm tập dữ liệu công khai: Apple (1.821 ảnh, 4 lớp), Embrapa (46.376 ảnh, 93 lớp), Maize (481 ảnh, 4 lớp), PlantVillage (54.305 ảnh, 38 lớp), và Rice (560 ảnh, 5 lớp).  

**Thí nghiệm và Kết quả:**  
- **Cấu hình thí nghiệm**: Mô hình được huấn luyện trên máy Nvidia DGX A100 với bốn GPU A100, sử dụng Keras, CUDA v11.5 và cuDNN v8.3. Ảnh được điều chỉnh kích thước thành 224x224x3, sử dụng hàm mất mát cross-entropy, bộ tối ưu Adam (tỷ lệ học 0,0001, kích thước batch 16).  
- **Hiệu suất**: PlantXViT đạt độ chính xác 93,55% (Apple), 89,24% (Embrapa), 92,59% (Maize), 98,86% (PlantVillage), và 98,33% (Rice), vượt trội so với các mô hình CNN khác [15, 16, 18-20].  
- **Tối ưu kích thước mảng (patch size)**: Kích thước mảng 5x5 cho kết quả tốt nhất về độ chính xác, độ chính xác (precision), độ nhớ (recall), và F1-score trên tất cả các tập dữ liệu (xem Bảng 4).  
- **Tối ưu bộ tối ưu (optimizer)**: Bộ tối ưu Adam cho hiệu suất ổn định nhất trên mọi tập dữ liệu, vượt trội so với SGD, RMSProp, Adamax, và Nadam (xem Bảng 5).  
- **Khả năng giải thích**: Sử dụng Grad-CAM và LIME để đánh giá khả năng giải thích. PlantXViT xác định chính xác vùng bệnh trên lá với độ chính xác cao hơn so với các mô hình khác. t-SNE plots cho thấy PlantXViT tạo ra các cụm đặc trưng riêng biệt, chứng minh khả năng trích xuất đặc trưng hiệu quả.  

**So sánh với các mô hình khác**: PlantXViT vượt trội so với các mô hình của Karthik et al. [15], Chen et al. [16, 18, 19], và Zhao et al. [20] về độ chính xác, precision, recall, F1-score, AUC, và kappa score trên cả năm tập dữ liệu.  

**Hạn chế**: Mô hình có số lượng phép tính dấu phẩy động (GFLOPs) cao hơn một số mô hình khác do khối Inception, nhưng bù lại có số tham số thấp (0,85 triệu), phù hợp với thiết bị IoT.

---

### **Kết luận (Conclusion)**  
**Nguyên văn:**  
In the present work, a ViT enabled CNN model is proposed for plant disease detection and identification... In future, it is planned to work on reducing the FLOPs count while maintaining the model's efficiency and explainability.

**Dịch sang tiếng Việt:**  
Nghiên cứu đề xuất mô hình PlantXViT, kết hợp CNN và Vision Transformer, để phát hiện và nhận diện bệnh cây trồng. Mô hình đạt độ chính xác cao (93,55%–98,86%) trên năm tập dữ liệu công khai, vượt trội so với các mô hình tiên tiến khác. Kết quả được đánh giá là dễ giải thích nhờ Grad-CAM và LIME. Với chỉ 0,85 triệu tham số, PlantXViT phù hợp cho các thiết bị IoT trong nông nghiệp thông minh. Tuy nhiên, hạn chế là yêu cầu tính toán (FLOPs) cao. Trong tương lai, nghiên cứu sẽ tập trung vào giảm FLOPs mà vẫn duy trì hiệu suất và khả năng giải thích của mô hình.

---

### **Tài liệu tham khảo (References)**  
**Nguyên văn:**  
[1] DESA. World population prospects 2019... [36] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. "Why should I trust you?"...

**Dịch sang tiếng Việt:**  
Tài liệu tham khảo bao gồm các nguồn từ báo cáo dân số thế giới (DESA, 2019), tiêu chuẩn FAO về sâu bệnh, và các nghiên cứu về nhận diện bệnh cây trồng sử dụng CNN, ViT, và cơ chế chú ý. Các tài liệu này được trích dẫn để hỗ trợ cơ sở lý thuyết và so sánh với các phương pháp hiện đại.
