# PDF Processing Tool - User Guide

## 🎯 Tổng quan

Tool tổng hợp để xử lý PDF một cách toàn diện:
- ✅ Tách PDF theo chapters (từ TOC hoặc pattern detection)
- ✅ Chia nhỏ mỗi chapter thành các file PDF 10 trang (hoặc tùy chỉnh)
- ✅ Extract ảnh từ mỗi chapter vào folder riêng
- ✅ Tự động filter icon/logo nhỏ
- ✅ Tạo metadata và README chi tiết

## 📦 Cài đặt

```bash
pip install PyMuPDF Pillow
```

## 🚀 Cách sử dụng cơ bản

### 1. Xử lý toàn diện (Recommended)

Tự động: extract chapters → split 10 trang/file → extract images

```bash
python pdf_tool.py book.pdf
```

**Output:**
```
book_processed/
├── chapter_1/
│   ├── pdfs/
│   │   ├── chapter_1_pages_1-10.pdf
│   │   ├── chapter_1_pages_11-20.pdf
│   │   └── chapter_1_pages_21-30.pdf
│   └── images/
│       ├── fig_1-1_diagram.png
│       ├── fig_1-2_example.png
│       └── images_metadata.json
├── chapter_2/
│   ├── pdfs/
│   └── images/
├── processing_summary.json
└── README.md
```

### 2. Custom output folder

```bash
python pdf_tool.py book.pdf -o my_output_folder
```

### 3. Thay đổi số trang mỗi file split

```bash
# Chia 20 trang mỗi file thay vì 10
python pdf_tool.py book.pdf --pages-per-file 20
```

### 4. Chỉ extract chapters (không split, không extract images)

```bash
python pdf_tool.py book.pdf --chapters-only
```

**Output:**
```
book_processed/
├── chapter_1/
│   └── chapter_1_full.pdf
├── chapter_2/
│   └── chapter_2_full.pdf
└── ...
```

### 5. Extract images nhưng không filter

```bash
# Lấy tất cả ảnh kể cả icon nhỏ
python pdf_tool.py book.pdf --no-filter-images
```

### 6. Tùy chỉnh size tối thiểu của ảnh

```bash
python pdf_tool.py book.pdf --min-width 150 --min-height 150
```

### 7. Custom pattern để detect chapters

```bash
# Nếu PDF không có TOC hoặc format chapter khác
python pdf_tool.py book.pdf --pattern "Chapter \\d+"
```

### 8. Include tất cả sections (Preface, Appendix, etc.)

```bash
python pdf_tool.py book.pdf --include-all
```

## 📊 Test Results

### Test 1: nlp-book.pdf (Full mode)

```bash
python pdf_tool.py nlp-book.pdf -o test_nlp_full --pages-per-file 10
```

**Kết quả:**
- ✅ 11 chapters detected
- ✅ 74 PDF chunks created (split 10 trang/file)
- ✅ 1 image extracted
- ✅ Total: 713 pages processed

**Cấu trúc:**
```
test_nlp_full/
├── chapter_1/ (6 PDF files, 0 images)
├── chapter_2/ (6 PDF files, 0 images)
├── chapter_3/ (5 PDF files, 0 images)
├── chapter_4/ (4 PDF files, 1 image)
├── chapter_5/ (6 PDF files, 0 images)
├── chapter_6/ (10 PDF files, 0 images)
├── chapter_7/ (4 PDF files, 0 images)
├── chapter_8/ (7 PDF files, 0 images)
├── chapter_9/ (7 PDF files, 0 images)
├── chapter_10/ (6 PDF files, 0 images)
├── chapter_11/ (13 PDF files, 0 images)
├── processing_summary.json
└── README.md
```

### Test 2: Prompt Engineering for Generative AI (Chapters-only mode)

```bash
python pdf_tool.py "Prompt Engineering for Generative AI..." \
  -o test_prompt_chapters --chapters-only
```

**Kết quả:**
- ✅ 10 chapters detected
- ✅ 121 images extracted
- ✅ Total: 423 pages processed

**Image distribution:**
- Chapter 1: 14 images
- Chapter 2: 5 images
- Chapter 3: 8 images
- Chapter 4: 6 images
- Chapter 5: 2 images
- Chapter 6: 5 images
- Chapter 7: 12 images
- Chapter 8: 31 images (nhiều nhất)
- Chapter 9: 33 images (nhiều nhất)
- Chapter 10: 5 images

## 🎨 Features chi tiết

### 1. Chapter Detection

Tool tự động detect chapters bằng:
- **TOC (Table of Contents)**: Ưu tiên đầu tiên
- **Pattern Matching**: Nếu không có TOC, tự động detect bằng patterns:
  - `Chapter \d+`
  - `CHAPTER \d+`
  - `Chương \d+`
  - `Part \d+`

### 2. Smart Image Filtering

- Tự động lọc bỏ icon/logo nhỏ (< 200x200px)
- Phát hiện figure captions tự động
- Đặt tên file có ý nghĩa: `fig_1-1_diagram_name.png`
- Phân loại ảnh: figure, diagram, screenshot, photo

### 3. Metadata

Mỗi lần xử lý tạo ra:
- `processing_summary.json`: Toàn bộ metadata chi tiết
- `README.md`: Documentation đầy đủ
- `images_metadata.json`: Metadata từng ảnh (trong mỗi chapter)
- Markdown index cho images (Obsidian-ready)

## 🔧 Advanced Options

### Full command với tất cả options

```bash
python pdf_tool.py book.pdf \
  -o output_folder \
  --pages-per-file 10 \
  --min-width 200 \
  --min-height 200 \
  --toc-level 1 \
  --pattern "Chapter \\d+" \
  --include-all \
  --no-filter-images \
  --chapters-only \
  --no-images
```

### Options reference

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `<pdf_name>_processed` | Output directory |
| `-p, --pages-per-file` | `10` | Số trang mỗi file khi split |
| `--chapters-only` | `False` | Chỉ extract chapters, không split |
| `--no-images` | `False` | Không extract images |
| `--no-filter-images` | `False` | Không filter icon/logo nhỏ |
| `--min-width` | `200` | Chiều rộng tối thiểu ảnh (px) |
| `--min-height` | `200` | Chiều cao tối thiểu ảnh (px) |
| `--toc-level` | `1` | Level của TOC để extract |
| `--pattern` | `None` | Custom regex pattern cho chapters |
| `--include-all` | `False` | Include tất cả TOC items |

## 📝 Use Cases

### Case 1: Học từ sách kỹ thuật

```bash
# Extract chapters và images để học từng chapter
python pdf_tool.py technical_book.pdf --chapters-only
```

→ Mỗi chapter 1 file PDF riêng + folder images riêng

### Case 2: Chia nhỏ để đọc trên mobile

```bash
# Split thành các file nhỏ, dễ đọc
python pdf_tool.py large_book.pdf --pages-per-file 5 --no-images
```

→ Mỗi file chỉ 5 trang, dễ load trên điện thoại

### Case 3: Extract tất cả diagrams/figures

```bash
# Chỉ extract images, không split
python pdf_tool.py textbook.pdf --chapters-only
```

→ Tất cả ảnh được tổ chức theo chapter

### Case 4: Prepare cho Obsidian vault

```bash
python pdf_tool.py book.pdf
```

→ Images có caption, markdown index, sẵn sàng import vào Obsidian

## 🐛 Troubleshooting

### PDF không detect được chapters

**Giải pháp:**
```bash
# Thử với custom pattern
python pdf_tool.py book.pdf --pattern "Chapter \\d+"

# Hoặc include tất cả sections
python pdf_tool.py book.pdf --include-all
```

### Quá nhiều icon/logo được extract

**Giải pháp:**
```bash
# Tăng min size
python pdf_tool.py book.pdf --min-width 300 --min-height 300
```

### Không muốn extract images

**Giải pháp:**
```bash
python pdf_tool.py book.pdf --no-images
```

## 📁 Output Structure Detail

```
book_processed/
├── chapter_1/
│   ├── pdfs/                          # PDF chunks
│   │   ├── chapter_1_pages_1-10.pdf
│   │   ├── chapter_1_pages_11-20.pdf
│   │   └── chapter_1_pages_21-25.pdf
│   └── images/                        # Images từ chapter này
│       ├── fig_1-1_diagram_name.png
│       ├── fig_1-2_example.png
│       ├── images_metadata.json       # Metadata từng ảnh
│       └── chapter_1_images.md        # Markdown index
├── chapter_2/
│   ├── pdfs/
│   └── images/
├── processing_summary.json            # Toàn bộ metadata
└── README.md                          # Auto-generated doc
```

## 🔗 Integration với các tools khác

### Với Obsidian

1. Copy images vào vault:
```bash
cp book_processed/chapter_*/images/*.png ~/ObsidianVault/attachments/
```

2. Import markdown index:
```bash
cp book_processed/chapter_*/images/*.md ~/ObsidianVault/
```

3. Embed trong notes:
```markdown
![[fig_1-1_diagram_name.png]]
```

### Với Anki (flashcards)

1. Extract chapters
2. Dùng PDF chunks để tạo cards theo chapter
3. Embed images từ folder images

## 📈 Performance

- **nlp-book.pdf** (713 pages): ~2 minutes (full mode)
- **Prompt Engineering PDF** (423 pages): ~1 minute (chapters-only)
- Memory usage: Low (streaming processing)

## ⚡ Tips & Best Practices

1. **Luôn dùng `--chapters-only` trước** để xem chapters có được detect đúng không
2. **Với sách có nhiều ảnh**, dùng `--min-width 250` để filter tốt hơn
3. **Kiểm tra `processing_summary.json`** để xem metadata chi tiết
4. **Dùng `--pattern`** nếu sách có format chapter đặc biệt
5. **Backup PDF gốc** trước khi xử lý

## 🎓 Examples

### Example 1: Technical Book

```bash
python pdf_tool.py "Machine Learning Book.pdf" \
  --pages-per-file 15 \
  --min-width 250
```

### Example 2: Novel (chỉ split, không extract images)

```bash
python pdf_tool.py "Novel.pdf" \
  --pages-per-file 20 \
  --no-images
```

### Example 3: Research Paper với nhiều diagrams

```bash
python pdf_tool.py "Research Paper.pdf" \
  --chapters-only \
  --no-filter-images \
  --include-all
```

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra dependencies: `pip install PyMuPDF Pillow`
2. Xem log trong terminal
3. Kiểm tra `processing_summary.json` để debug

---

**Created by:** PDF Processing Tool v1.0  
**Date:** January 2026
