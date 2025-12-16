#!/usr/bin/env python3
"""
PDF Image Extractor for Obsidian
Trích xuất ảnh từ PDF, lọc icon/logo, tối ưu cho Obsidian notes

QUICK USAGE:
    # Basic - extract vào thư mục cùng tên với PDF
    python pdf_image_extractor.py book.pdf
    # → Output: book/ (cùng thư mục với book.pdf)

    # Custom output folder
    python pdf_image_extractor.py book.pdf -o my_images

    # Điều chỉnh filter (tăng threshold để lọc nhiều hơn)
    python pdf_image_extractor.py book.pdf -w 250 --min-height 250

    # Lấy tất cả ảnh (không filter)
    python pdf_image_extractor.py book.pdf --no-filter

    # Không tạo markdown index
    python pdf_image_extractor.py book.pdf --no-markdown

FEATURES:
    ✅ Lọc icon/logo nhỏ tự động
    ✅ Trích xuất caption từ PDF
    ✅ Phân loại: figure, diagram, screenshot, photo
    ✅ Tên file có ý nghĩa: fig_6-4_babyagi_architecture.png
    ✅ Markdown index tự động cho Obsidian
    ✅ Metadata JSON đầy đủ

OUTPUT:
    # Mặc định: thư mục cùng tên với PDF
    book.pdf → book/
    ├── fig_1-1_tokens_breakdown.png
    ├── fig_6-4_babyagi_architecture.png
    ├── images_metadata.json
    └── book_images.md

REQUIREMENTS:
    pip install PyMuPDF Pillow

For full help:
    python pdf_image_extractor.py --help
"""

import fitz  # PyMuPDF
import os
import sys
import json
import re
import argparse
from pathlib import Path
from PIL import Image
import io


def is_figure_caption(text):
    """Kiểm tra xem text có phải là caption của figure/diagram không"""
    if not text:
        return False

    patterns = [
        r"^\s*Figure\s+\d+[-. ]\d+",
        r"^\s*Fig\.?\s+\d+[-. ]\d+",
        r"^\s*Diagram\s+\d+[-. ]\d+",
        r"^\s*\d+[-. ]\d+\.",  # "6-4." pattern
        r"^\s*Image\s+\d+",
    ]

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def find_image_caption(page, img_bbox, search_range=150):
    """Tìm caption/title của ảnh xung quanh vị trí ảnh"""
    x0, y0, x1, y1 = img_bbox

    # Mở rộng vùng tìm kiếm
    search_rect_below = fitz.Rect(x0 - 100, y1, x1 + 100, y1 + search_range)
    search_rect_above = fitz.Rect(x0 - 100, y0 - search_range, x1 + 100, y0)

    text_below = page.get_text("text", clip=search_rect_below).strip()
    text_above = page.get_text("text", clip=search_rect_above).strip()

    # Patterns mở rộng cho figure captions
    patterns = [
        r"Figure\s+\d+[-. ]\d+[:.]\s*(.+)",
        r"Fig\.?\s*\d+[-. ]\d+[:.]\s*(.+)",
        r"\d+[-. ]\d+\.\s+(.+)",  # "6-4. BabyAGI's agent architecture"
        r"Figure\s+\d+[:.]\s*(.+)",
        r"Fig\.\s*\d+[:.]\s*(.+)",
        r"Image\s+\d+[:.]\s*(.+)",
        r"Diagram\s+\d+[:.]\s*(.+)",
    ]

    # Tìm caption phía dưới (phổ biến nhất)
    for pattern in patterns:
        match = re.search(pattern, text_below, re.IGNORECASE | re.MULTILINE)
        if match:
            lines = text_below[match.start() :].split("\n")
            caption_parts = []
            for line in lines[:3]:  # Lấy tối đa 3 dòng
                line = line.strip()
                if line and not line.startswith(("Page", "Chapter", "|", "[")):
                    caption_parts.append(line)
                    if len(" ".join(caption_parts)) > 120:
                        break
                else:
                    break
            if caption_parts:
                return " ".join(caption_parts)

    # Tìm phía trên
    for pattern in patterns:
        match = re.search(pattern, text_above, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(0).strip()

    return None


def classify_image(width, height, size_bytes, caption):
    """Phân loại ảnh: figure, diagram, icon, decorative"""
    # Icon/Logo nhỏ - cần loại bỏ
    if (width < 150 and height < 150) or size_bytes < 25000:
        return "icon"

    # Ảnh quá nhỏ
    if width < 200 and height < 200:
        return "decorative"

    # Figure/Diagram dựa vào caption
    if caption and is_figure_caption(caption):
        caption_lower = caption.lower()
        if any(
            word in caption_lower
            for word in [
                "architecture",
                "flow",
                "diagram",
                "process",
                "pipeline",
                "framework",
            ]
        ):
            return "diagram"
        elif any(
            word in caption_lower for word in ["screenshot", "interface", "ui", "web"]
        ):
            return "screenshot"
        elif any(word in caption_lower for word in ["photo", "picture"]):
            return "photo"
        return "figure"

    # Dựa vào kích thước
    if 300 < width < 800 and 200 < height < 600:
        return "diagram"

    if width > 500 and height > 400:
        return "screenshot"

    return "figure"


def extract_figure_number(caption):
    """Trích xuất số hiệu figure: 'Figure 6-4. BabyAGI' -> '6-4'"""
    if not caption:
        return None

    patterns = [
        r"Figure\s+(\d+[-. ]\d+)",
        r"Fig\.?\s+(\d+[-. ]\d+)",
        r"^(\d+[-. ]\d+)\.",
    ]

    for pattern in patterns:
        match = re.search(pattern, caption, re.IGNORECASE)
        if match:
            return match.group(1).replace(" ", "-").replace(".", "-")

    return None


def sanitize_filename(text):
    """Làm sạch text để dùng làm tên file"""
    if not text:
        return ""

    # Loại bỏ ký tự không hợp lệ
    text = re.sub(r'[<>:"/\\|?*]', "", text)
    # Thay thế khoảng trắng
    text = re.sub(r"\s+", "_", text)
    # Loại bỏ dấu câu thừa
    text = re.sub(r"[.]+$", "", text)
    text = re.sub(r"_+", "_", text)
    # Loại bỏ ký tự đặc biệt
    text = re.sub(r"[" '"""]', "", text)

    return text.strip("_")[:120]


def create_obsidian_markdown_index(images_metadata, output_folder, pdf_name):
    """Tạo file markdown index cho Obsidian"""
    md_content = f"# Images from {pdf_name}\n\n"
    md_content += f"Extracted {len(images_metadata)} images\n\n"
    md_content += "---\n\n"

    # Group by page
    pages = {}
    for img in images_metadata:
        page = img["page"]
        if page not in pages:
            pages[page] = []
        pages[page].append(img)

    # Generate markdown
    for page in sorted(pages.keys()):
        md_content += f"## Page {page}\n\n"
        for img in pages[page]:
            md_content += f"### {img.get('caption', 'Untitled Image')}\n\n"
            md_content += f"![[{img['filename']}]]\n\n"
            md_content += f"- **Type**: {img['type']}\n"
            md_content += f"- **Size**: {img['width']}x{img['height']} px\n"
            if img.get("figure_number"):
                md_content += f"- **Figure**: {img['figure_number']}\n"
            md_content += "\n"

    # Save markdown file
    md_path = Path(output_folder) / f"{sanitize_filename(pdf_name)}_images.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return md_path


def pdf_image_extractor(
    pdf_path,
    output_folder="obsidian_images",
    min_width=200,
    min_height=200,
    filter_icons=True,
    create_markdown=True,
):
    """
    Trích xuất ảnh từ PDF, tối ưu cho Obsidian

    Args:
        pdf_path: Đường dẫn PDF
        output_folder: Thư mục output
        min_width: Chiều rộng tối thiểu (filter icon)
        min_height: Chiều cao tối thiểu (filter icon)
        filter_icons: Có lọc icon/logo không
        create_markdown: Có tạo markdown index không

    Returns:
        dict với 'images': metadata list, 'markdown_file': đường dẫn md
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    pdf_document = fitz.open(pdf_path)
    pdf_name = Path(pdf_path).stem

    all_images_metadata = []
    image_count = 0
    filtered_count = 0

    print(f"{'='*70}")
    print(f"OBSIDIAN PDF IMAGE EXTRACTOR")
    print(f"{'='*70}")
    print(f"📄 PDF: {pdf_name}")
    print(f"📁 Output: {output_folder}")
    print(f"🔍 Filter icons: {filter_icons}")
    print(f"📏 Min size: {min_width}x{min_height} px\n")

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)

        if len(image_list) == 0:
            continue

        print(f"\n📃 Page {page_num + 1}: Found {len(image_list)} images")

        for img_index, img in enumerate(image_list):
            xref = img[0]

            # Extract image
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Get dimensions
            try:
                pil_image = Image.open(io.BytesIO(image_bytes))
                width, height = pil_image.size
            except:
                width, height = 0, 0
                continue

            # Find caption
            img_rects = page.get_image_rects(xref)
            caption = None
            if img_rects:
                img_bbox = img_rects[0]
                caption = find_image_caption(page, img_bbox)

            # Classify
            img_type = classify_image(width, height, len(image_bytes), caption)

            # Filter icons/small images
            if filter_icons and img_type in ["icon", "decorative"]:
                filtered_count += 1
                print(f"  ⊘ Skipped [{img_type}]: {width}x{height}px")
                continue

            if width < min_width or height < min_height:
                filtered_count += 1
                print(f"  ⊘ Skipped [too small]: {width}x{height}px")
                continue

            # Create filename optimized for Obsidian
            fig_num = extract_figure_number(caption)

            if fig_num:
                # Example: fig_6-4_babyagi_architecture.png
                desc = caption.split(".", 1)[-1] if "." in caption else caption
                desc = sanitize_filename(desc)[:50]
                filename = f"fig_{fig_num}_{desc}.{image_ext}"
            elif caption:
                desc = sanitize_filename(caption)[:60]
                filename = f"p{page_num + 1}_{desc}.{image_ext}"
            else:
                filename = f"p{page_num + 1}_img{image_count + 1}.{image_ext}"

            # Save image
            img_path = output_path / filename
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            # Store metadata
            metadata = {
                "image_id": image_count + 1,
                "filename": filename,
                "page": page_num + 1,
                "caption": caption,
                "width": width,
                "height": height,
                "format": image_ext,
                "size_bytes": len(image_bytes),
                "type": img_type,
                "figure_number": fig_num,
            }
            all_images_metadata.append(metadata)

            # Print info
            print(f"  ✓ [{img_type}] {filename}")
            if caption:
                print(f"     {caption[:80]}...")

            image_count += 1

    pdf_document.close()

    # Save JSON metadata
    json_path = output_path / "images_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_images_metadata, f, ensure_ascii=False, indent=2)

    # Create Markdown index
    md_path = None
    if create_markdown and all_images_metadata:
        md_path = create_obsidian_markdown_index(
            all_images_metadata, output_folder, pdf_name
        )
        print(f"\n📝 Markdown index: {md_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"✅ COMPLETED")
    print(f"{'='*70}")
    print(f"📊 Extracted: {image_count} images")
    print(f"🗑️  Filtered: {filtered_count} icons/small images")
    print(f"📁 Output folder: {output_folder}")
    print(f"📋 Metadata: {json_path}")
    if md_path:
        print(f"📝 Markdown: {md_path}")
    print(f"{'='*70}\n")

    return {
        "images": all_images_metadata,
        "markdown_file": str(md_path) if md_path else None,
        "total_extracted": image_count,
        "total_filtered": filtered_count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract images from PDF, optimized for Obsidian notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s book.pdf
    → Output: book/ (same directory as PDF)
  
  %(prog)s /path/to/my_book.pdf -o custom_folder
    → Output: custom_folder/
  
  %(prog)s book.pdf --min-width 150 --min-height 150
  %(prog)s book.pdf --no-filter --no-markdown

Output:
  Images will be saved with descriptive filenames like:
  - fig_6-4_babyagi_architecture.png (with figure number)
  - p280_diagram_name.png (without figure number)
  
  A markdown index file will be created for easy navigation in Obsidian.
        """,
    )

    parser.add_argument("pdf_file", help="Path to the PDF file to extract images from")

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output folder (default: same directory as PDF, named after PDF file)",
    )

    parser.add_argument(
        "-w",
        "--min-width",
        type=int,
        default=200,
        help="Minimum image width in pixels to extract (default: 200)",
    )

    parser.add_argument(
        "--min-height",
        type=int,
        default=200,
        help="Minimum image height in pixels to extract (default: 200)",
    )

    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable icon/logo filtering (extract all images)",
    )

    parser.add_argument(
        "--no-markdown", action="store_true", help="Don't create markdown index file"
    )

    parser.add_argument(
        "-r",
        "--search-range",
        type=int,
        default=150,
        help="Caption search range in pixels (default: 150)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate PDF file
    if not os.path.exists(args.pdf_file):
        print(f"❌ Error: File not found: {args.pdf_file}")
        sys.exit(1)

    if not args.pdf_file.lower().endswith(".pdf"):
        print(f"⚠️  Warning: File doesn't have .pdf extension: {args.pdf_file}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            sys.exit(0)

    # Run extraction
    try:
        # Determine output folder
        if args.output is None:
            # Mặc định: thư mục cùng cấp, tên theo PDF
            pdf_path = Path(args.pdf_file)
            output_folder = pdf_path.parent / pdf_path.stem
        else:
            output_folder = args.output

        result = pdf_image_extractor(
            pdf_path=args.pdf_file,
            output_folder=str(output_folder),
            min_width=args.min_width,
            min_height=args.min_height,
            filter_icons=not args.no_filter,
            create_markdown=not args.no_markdown,
        )

        # Success message
        print(f"\n{'='*70}")
        print("🎯 Next steps for Obsidian:")
        print(f"{'='*70}")
        print(f"1. Copy images to your Obsidian vault:")
        print(f"   cp {output_folder}/*.png ~/ObsidianVault/attachments/")
        print(f"\n2. Import the markdown file:")
        if result["markdown_file"]:
            print(f"   cp {result['markdown_file']} ~/ObsidianVault/")
        print(f"\n3. Use [[filename]] syntax to embed images in your notes")
        print(f"\n📊 Summary:")
        print(f"   • Extracted: {result['total_extracted']} images")
        print(f"   • Filtered: {result['total_filtered']} icons/decorative")
        print(f"   • Output: {output_folder}/")

    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
