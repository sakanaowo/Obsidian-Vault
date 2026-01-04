#!/usr/bin/env python3
"""
PDF Chapter Extractor
Tách PDF theo từng chương thực sự (Chapter 1, 2, 3...) từ TOC hoặc pattern detection

QUICK USAGE:
    # Basic - tách theo chapters thực sự trong TOC
    python pdf_chapter_extractor.py book.pdf

    # Custom output folder
    python pdf_chapter_extractor.py book.pdf -o chapters_output

    # Bao gồm cả các phần như Preface, Appendix (không chỉ chapters)
    python pdf_chapter_extractor.py book.pdf --include-all

    # Sử dụng pattern detection thay vì TOC
    python pdf_chapter_extractor.py book.pdf --pattern "Chapter \\d+"

    # Liệt kê chapters mà không extract
    python pdf_chapter_extractor.py book.pdf --list-only

FEATURES:
    ✅ Tự động detect và lọc chỉ các chapters thực sự (Chapter X)
    ✅ Bỏ qua Cover, Copyright, TOC, Index... 
    ✅ Đánh số theo số chapter gốc (Chapter 1 → chapter_1.pdf)
    ✅ Hỗ trợ pattern matching cho các PDF không có TOC
    ✅ Tích hợp với pdf_image_extractor.py

OUTPUT:
    book.pdf → book_chapters/
    ├── book_chapter_1.pdf   (Chapter 1)
    ├── book_chapter_2.pdf   (Chapter 2)
    ├── book_chapter_3.pdf   (Chapter 3)
    └── chapters_info.json

REQUIREMENTS:
    pip install PyMuPDF

For full help:
    python pdf_chapter_extractor.py --help
"""

import fitz  # PyMuPDF
import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def sanitize_filename(name: str, max_length: int = 50) -> str:
    """Làm sạch tên file, loại bỏ ký tự không hợp lệ"""
    # Loại bỏ ký tự không hợp lệ cho filename
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # Thay thế khoảng trắng liên tiếp bằng underscore
    name = re.sub(r'\s+', '_', name.strip())
    # Giới hạn độ dài
    if len(name) > max_length:
        name = name[:max_length]
    return name


def extract_chapter_number(title: str) -> Optional[int]:
    """
    Trích xuất số chapter từ title
    Ví dụ: "Chapter 1. Introduction" → 1
           "Chapter 10: Advanced Topics" → 10
           "CHAPTER 5 - Methods" → 5
    """
    patterns = [
        r'Chapter\s+(\d+)',
        r'CHAPTER\s+(\d+)',
        r'Chương\s+(\d+)',
        r'CHƯƠNG\s+(\d+)',
        r'Part\s+(\d+)',
        r'PART\s+(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return None


def is_real_chapter(title: str) -> bool:
    """
    Kiểm tra xem title có phải là chapter thực sự không
    Loại bỏ các mục như: Cover, Copyright, TOC, Index, Preface, About, Colophon...
    """
    # Các patterns cho chapters thực sự
    chapter_patterns = [
        r'Chapter\s+\d+',
        r'CHAPTER\s+\d+',
        r'Chương\s+\d+',
        r'Part\s+\d+',
        r'PART\s+\d+',
    ]
    
    for pattern in chapter_patterns:
        if re.search(pattern, title, re.IGNORECASE):
            return True
    
    return False


def is_supplementary_section(title: str) -> bool:
    """
    Kiểm tra xem có phải là phần bổ sung cần bỏ qua không
    """
    skip_patterns = [
        r'^Cover$',
        r'^Copyright',
        r'^Table of Contents',
        r'^Contents$',
        r'^Index$',
        r'^Preface',
        r'^Foreword',
        r'^Introduction$',  # Chỉ "Introduction" đứng một mình, không phải "Chapter 1: Introduction"
        r'^About the Author',
        r'^Acknowledgment',
        r'^Colophon',
        r'^Appendix',
        r'^Bibliography',
        r'^References$',
        r'^Glossary',
        r'^Dedication',
    ]
    
    title_clean = title.strip()
    
    for pattern in skip_patterns:
        if re.search(pattern, title_clean, re.IGNORECASE):
            return True
    
    return False


def get_toc_chapters(doc: fitz.Document, level: int = 1, include_all: bool = False) -> List[Dict]:
    """
    Lấy danh sách chapters từ Table of Contents của PDF
    
    Args:
        doc: PyMuPDF document
        level: Level của TOC cần lấy (1 = top level chapters)
        include_all: Nếu True, bao gồm tất cả mục; False chỉ lấy chapters thực sự
    
    Returns:
        List of chapter info dicts với keys: title, start_page, end_page, original_chapter_num
    """
    toc = doc.get_toc()
    
    if not toc:
        return []
    
    total_pages = len(doc)
    
    # Lấy tất cả các mục ở level chỉ định
    all_items = [(title, page) for lvl, title, page in toc if lvl == level]
    
    if not all_items:
        return []
    
    # Lọc chỉ các chapters thực sự (nếu không include_all)
    if include_all:
        filtered_items = all_items
    else:
        filtered_items = [(title, page) for title, page in all_items if is_real_chapter(title)]
    
    if not filtered_items:
        return []
    
    chapters = []
    
    for i, (title, start_page) in enumerate(filtered_items):
        # Tìm end_page: là start của item tiếp theo trong all_items (không phải filtered)
        # để đảm bảo lấy đủ nội dung
        current_idx = next((idx for idx, (t, p) in enumerate(all_items) if t == title and p == start_page), -1)
        
        if current_idx >= 0 and current_idx + 1 < len(all_items):
            end_page = all_items[current_idx + 1][1] - 1
        else:
            end_page = total_pages
        
        # Điều chỉnh page numbers
        start_page = max(1, start_page)
        end_page = min(total_pages, end_page)
        
        # Lấy số chapter gốc từ title
        original_num = extract_chapter_number(title)
        
        chapters.append({
            'chapter_num': original_num if original_num else (i + 1),
            'title': title,
            'start_page': start_page,
            'end_page': end_page,
            'page_count': end_page - start_page + 1
        })
    
    return chapters


def detect_chapters_by_pattern(doc: fitz.Document, pattern: str = r"Chapter\s+\d+") -> List[Dict]:
    """
    Detect chapters bằng pattern matching trong text của từng trang
    Chỉ detect trang bắt đầu chapter mới (không phải header/footer)
    
    Args:
        doc: PyMuPDF document
        pattern: Regex pattern để match chapter headings
    
    Returns:
        List of chapter info dicts
    """
    total_pages = len(doc)
    chapter_pages = []
    seen_chapters = set()  # Để tránh trùng lặp
    
    regex = re.compile(pattern, re.IGNORECASE)
    
    for page_num in range(total_pages):
        page = doc[page_num]
        
        # Lấy text blocks với thông tin vị trí
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        
        page_height = page.rect.height
        page_width = page.rect.width
        
        for block in blocks:
            if block["type"] != 0:  # Chỉ xử lý text blocks
                continue
            
            # Bỏ qua header (top 10%) và footer (bottom 10%)
            block_y = block.get("bbox", [0, 0, 0, 0])[1]
            if block_y < page_height * 0.1 or block_y > page_height * 0.85:
                continue
            
            # Lấy text từ block
            block_text = ""
            max_font_size = 0
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
                    font_size = span.get("size", 0)
                    if font_size > max_font_size:
                        max_font_size = font_size
            
            block_text = block_text.strip()
            
            # Kiểm tra xem block có match chapter pattern không
            match = regex.search(block_text)
            if match:
                # Kiểm tra xem đây có phải là chapter heading thực sự không:
                # 1. Font size lớn (> 14pt) hoặc
                # 2. Text bắt đầu với "Chapter X" (không phải reference trong văn bản)
                # 3. Block nằm ở phần trên của trang (< 40% chiều cao)
                
                is_heading = False
                
                # Check font size lớn
                if max_font_size >= 14:
                    is_heading = True
                
                # Check text bắt đầu với Chapter pattern
                if re.match(rf"^\s*{pattern}", block_text, re.IGNORECASE):
                    # Và nằm ở phần trên trang
                    if block_y < page_height * 0.4:
                        is_heading = True
                
                # Check độ dài text - chapter heading thường ngắn
                if len(block_text) < 100 and re.match(rf"^\s*{pattern}", block_text, re.IGNORECASE):
                    is_heading = True
                
                if is_heading:
                    chapter_num = extract_chapter_number(block_text)
                    
                    # Chỉ thêm nếu chưa có chapter này (tránh trùng lặp từ header/footer)
                    if chapter_num and chapter_num not in seen_chapters:
                        seen_chapters.add(chapter_num)
                        
                        # Lấy title đầy đủ
                        title_match = re.search(rf"({pattern}[^\n]*)", block_text, re.IGNORECASE)
                        title = title_match.group(1).strip() if title_match else match.group(0)
                        
                        chapter_pages.append({
                            'title': title,
                            'page': page_num + 1,
                            'chapter_num': chapter_num
                        })
                        break  # Đã tìm thấy chapter heading trong trang này
    
    # Sắp xếp theo chapter number
    chapter_pages.sort(key=lambda x: (x['chapter_num'], x['page']))
    
    # Loại bỏ trùng lặp chapter number (giữ lại entry đầu tiên)
    unique_chapters = []
    seen = set()
    for ch in chapter_pages:
        if ch['chapter_num'] not in seen:
            seen.add(ch['chapter_num'])
            unique_chapters.append(ch)
    
    chapter_pages = unique_chapters
    
    # Tạo chapter ranges
    chapters = []
    for i, ch in enumerate(chapter_pages):
        if i + 1 < len(chapter_pages):
            end_page = chapter_pages[i + 1]['page'] - 1
        else:
            end_page = total_pages
        
        chapters.append({
            'chapter_num': ch['chapter_num'],
            'title': ch['title'],
            'start_page': ch['page'],
            'end_page': end_page,
            'page_count': end_page - ch['page'] + 1
        })
    
    return chapters


def extract_chapter(doc: fitz.Document, start_page: int, end_page: int, output_path: str) -> bool:
    """
    Extract một range của pages thành PDF mới
    
    Args:
        doc: Source PDF document
        start_page: Start page (1-indexed)
        end_page: End page (1-indexed, inclusive)
        output_path: Path to save extracted PDF
    
    Returns:
        True if successful
    """
    try:
        # Tạo document mới
        new_doc = fitz.open()
        
        # Insert pages (0-indexed)
        new_doc.insert_pdf(doc, from_page=start_page - 1, to_page=end_page - 1)
        
        # Save
        new_doc.save(output_path)
        new_doc.close()
        
        return True
    except Exception as e:
        print(f"Error extracting pages {start_page}-{end_page}: {e}")
        return False


def extract_chapters(
    pdf_path: str,
    output_dir: Optional[str] = None,
    level: int = 1,
    pattern: Optional[str] = None,
    list_only: bool = False,
    include_all: bool = False
) -> Dict:
    """
    Main function để extract chapters từ PDF
    
    Args:
        pdf_path: Path to source PDF
        output_dir: Output directory (default: <pdf_name>_chapters)
        level: TOC level to extract
        pattern: Custom pattern for chapter detection
        list_only: Only list chapters, don't extract
        include_all: Include all TOC items, not just chapters
    
    Returns:
        Dict with extraction results
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Tạo output directory name
    doc_name = pdf_path.stem
    # Rút gọn tên nếu quá dài
    short_name = sanitize_filename(doc_name, max_length=40)
    
    if output_dir is None:
        output_dir = pdf_path.parent / f"{short_name}_chapters"
    else:
        output_dir = Path(output_dir)
    
    # Open PDF
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    
    print(f"📖 Processing: {pdf_path.name}")
    print(f"   Total pages: {total_pages}")
    
    # Detect chapters
    if pattern:
        print(f"   Using pattern: {pattern}")
        chapters = detect_chapters_by_pattern(doc, pattern)
    else:
        print(f"   Using TOC (level {level})")
        if include_all:
            print("   Mode: Include ALL sections")
        else:
            print("   Mode: Only real chapters (Chapter X)")
        
        chapters = get_toc_chapters(doc, level, include_all)
        
        # Nếu không có TOC hoặc không tìm thấy chapters, thử detect bằng patterns
        if not chapters:
            print("   No chapters found in TOC, trying pattern detection...")
            common_patterns = [
                r"Chapter\s+\d+",
                r"CHAPTER\s+\d+",
                r"Chương\s+\d+",
                r"Part\s+\d+",
                r"PART\s+\d+",
            ]
            
            for p in common_patterns:
                chapters = detect_chapters_by_pattern(doc, p)
                if chapters:
                    print(f"   Found chapters using pattern: {p}")
                    break
    
    if not chapters:
        print("❌ No chapters detected!")
        print("   Try using --pattern to specify a custom pattern")
        print("   Or use --include-all to include all TOC items")
        doc.close()
        return {'success': False, 'error': 'No chapters detected'}
    
    # List chapters
    print(f"\n📑 Found {len(chapters)} chapters:")
    print("-" * 70)
    for ch in chapters:
        title_display = ch['title'][:50]
        print(f"   Chapter {ch['chapter_num']:2d}: {title_display:<50} (p.{ch['start_page']}-{ch['end_page']})")
    print("-" * 70)
    
    if list_only:
        doc.close()
        return {'success': True, 'chapters': chapters, 'extracted': False}
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Extract chapters
    extracted_files = []
    
    for ch in chapters:
        # Format: <doc_name>_chapter_<x>.pdf - sử dụng số chapter gốc
        output_filename = f"{short_name}_chapter_{ch['chapter_num']}.pdf"
        output_path = output_dir / output_filename
        
        print(f"   Extracting Chapter {ch['chapter_num']}: {ch['title'][:40]}...")
        
        success = extract_chapter(
            doc, 
            ch['start_page'], 
            ch['end_page'], 
            str(output_path)
        )
        
        if success:
            extracted_files.append({
                'filename': output_filename,
                'path': str(output_path),
                'chapter_num': ch['chapter_num'],
                'title': ch['title'],
                'start_page': ch['start_page'],
                'end_page': ch['end_page'],
                'page_count': ch['page_count']
            })
    
    doc.close()
    
    # Save metadata
    metadata = {
        'source_pdf': str(pdf_path),
        'source_name': doc_name,
        'total_pages': total_pages,
        'total_chapters': len(chapters),
        'output_dir': str(output_dir),
        'chapters': extracted_files
    }
    
    metadata_path = output_dir / 'chapters_info.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Extracted {len(extracted_files)} chapters to {output_dir}")
    print(f"   Metadata saved to: {metadata_path.name}")
    
    return {
        'success': True,
        'output_dir': str(output_dir),
        'chapters': extracted_files,
        'metadata_file': str(metadata_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract chapters from PDF based on TOC or pattern detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s book.pdf                    # Extract only real chapters (Chapter 1, 2, 3...)
  %(prog)s book.pdf --include-all      # Include all sections (Preface, Appendix, etc.)
  %(prog)s book.pdf -o my_chapters     # Custom output dir
  %(prog)s book.pdf --pattern "Chapter \\d+"  # Custom pattern
  %(prog)s book.pdf --list-only        # List chapters without extracting

Integration with pdf_image_extractor.py:
  After extracting chapters, you can process each chapter:
  
  for chapter in chapters/*.pdf; do
    python pdf_image_extractor.py "$chapter"
  done
        """
    )
    
    parser.add_argument('pdf', help='Input PDF file path')
    parser.add_argument('-o', '--output', help='Output directory (default: <pdf_name>_chapters)')
    parser.add_argument('--level', type=int, default=1, 
                        help='TOC level to extract (default: 1 for top-level)')
    parser.add_argument('--pattern', help='Custom regex pattern to detect chapters')
    parser.add_argument('--list-only', action='store_true',
                        help='Only list detected chapters, do not extract')
    parser.add_argument('--include-all', action='store_true',
                        help='Include all TOC sections (Cover, Preface, Appendix...), not just chapters')
    
    args = parser.parse_args()
    
    try:
        result = extract_chapters(
            args.pdf,
            output_dir=args.output,
            level=args.level,
            pattern=args.pattern,
            list_only=args.list_only,
            include_all=args.include_all
        )
        
        if result['success']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
