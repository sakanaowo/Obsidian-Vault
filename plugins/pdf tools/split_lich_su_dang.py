#!/usr/bin/env python3
"""
Script tách PDF Giáo trình Lịch sử Đảng Cộng sản Việt Nam theo chương
Sử dụng PyMuPDF để tìm và tách các chương
"""

import fitz  # PyMuPDF
import os
import sys
import json
import re
from pathlib import Path


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """Làm sạch tên file"""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name[:max_length]


def detect_chuong_pages(pdf_path: str) -> list:
    """
    Phát hiện các trang bắt đầu của mỗi chương
    Pattern: "Chương nhập môn" hoặc "Chương X" (X là số)
    """
    doc = fitz.open(pdf_path)
    chapters = []
    seen_chapters = set()  # Để tránh trùng lặp
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        
        page_height = page.rect.height
        
        for block in blocks:
            if block["type"] != 0:  # Chỉ xử lý text blocks
                continue
            
            # Bỏ qua footer (bottom 15%)
            block_y = block.get("bbox", [0, 0, 0, 0])[1]
            if block_y > page_height * 0.85:
                continue
            
            # Lấy text và font size
            block_text = ""
            max_font_size = 0
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
                    font_size = span.get("size", 0)
                    if font_size > max_font_size:
                        max_font_size = font_size
            
            block_text = block_text.strip()
            
            # Kiểm tra font size lớn (heading)
            if max_font_size < 12:
                continue
            
            # Kiểm tra "Chương nhập môn"
            if re.match(r'^Chương\s+nhập\s+môn', block_text, re.IGNORECASE):
                if 'nhap_mon' not in seen_chapters:
                    seen_chapters.add('nhap_mon')
                    chapters.append({
                        'title': 'Chương nhập môn',
                        'page': page_num + 1,
                        'chapter_num': 0
                    })
                    break
            
            # Kiểm tra "Chương X" (X là số)
            match = re.match(r'^Chương\s+(\d+)', block_text)
            if match:
                chapter_num = int(match.group(1))
                # Tránh trùng lặp
                if chapter_num not in seen_chapters:
                    seen_chapters.add(chapter_num)
                    chapters.append({
                        'title': f'Chương {chapter_num}',
                        'page': page_num + 1,
                        'chapter_num': chapter_num
                    })
                    break
    
    doc.close()
    
    # Sắp xếp theo chapter number
    chapters.sort(key=lambda x: x['chapter_num'])
    
    return chapters


def extract_chapters(pdf_path: str, output_dir: str):
    """Tách PDF theo các chương"""
    
    print(f"\n{'='*70}")
    print("TÁCH CHƯƠNG - GIÁO TRÌNH LỊCH SỬ ĐẢNG CỘNG SẢN VIỆT NAM")
    print(f"{'='*70}")
    
    # Tạo output directory
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"📖 PDF: {os.path.basename(pdf_path)}")
    print(f"📄 Total pages: {total_pages}")
    print(f"📁 Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Detect chapters
    print("🔍 Đang phát hiện các chương...")
    chapters = detect_chuong_pages(pdf_path)
    
    if not chapters:
        print("❌ Không tìm thấy chương nào!")
        return
    
    print(f"✅ Tìm thấy {len(chapters)} chương\n")
    
    # Tính toán page range cho mỗi chương
    for i, ch in enumerate(chapters):
        if i + 1 < len(chapters):
            ch['end_page'] = chapters[i + 1]['page'] - 1
        else:
            ch['end_page'] = total_pages
        ch['page_count'] = ch['end_page'] - ch['page'] + 1
    
    # In danh sách chương
    print("📑 Chi tiết các chương:")
    print("-" * 70)
    for ch in chapters:
        print(f"   {ch['title']:30} (trang {ch['page']:3} - {ch['end_page']:3}, {ch['page_count']} trang)")
    print("-" * 70 + "\n")
    
    # Tách từng chương
    print("⚙️  Đang tách các chương...\n")
    
    chapters_info = []
    
    for ch in chapters:
        # Tạo tên thư mục
        if ch['chapter_num'] == 0:
            dir_name = "chuong_nhap_mon"
        else:
            dir_name = f"chuong_{ch['chapter_num']}"
        
        chapter_dir = os.path.join(output_dir, dir_name)
        os.makedirs(chapter_dir, exist_ok=True)
        
        # Tách PDF
        pdf_name = f"{dir_name}.pdf"
        pdf_path_out = os.path.join(chapter_dir, pdf_name)
        
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=ch['page']-1, to_page=ch['end_page']-1)
        new_doc.save(pdf_path_out)
        new_doc.close()
        
        print(f"   ✓ {ch['title']}: {pdf_name} ({ch['page_count']} trang)")
        
        chapters_info.append({
            'title': ch['title'],
            'chapter_num': ch['chapter_num'],
            'directory': dir_name,
            'pdf_file': pdf_name,
            'start_page': ch['page'],
            'end_page': ch['end_page'],
            'page_count': ch['page_count']
        })
    
    doc.close()
    
    # Lưu metadata
    metadata_path = os.path.join(output_dir, 'chapters_info.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source_pdf': os.path.basename(pdf_path),
            'total_pages': total_pages,
            'total_chapters': len(chapters_info),
            'chapters': chapters_info
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print("✅ HOÀN THÀNH!")
    print(f"{'='*70}")
    print(f"📊 Tổng kết:")
    print(f"   • Số chương: {len(chapters_info)}")
    print(f"   • Output: {output_dir}")
    print(f"   • Metadata: chapters_info.json")
    print(f"{'='*70}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Tách PDF Giáo trình Lịch sử Đảng theo chương'
    )
    parser.add_argument('pdf_path', help='Đường dẫn tới file PDF')
    parser.add_argument('-o', '--output', help='Thư mục output', default=None)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"❌ Không tìm thấy file: {args.pdf_path}")
        sys.exit(1)
    
    # Default output directory
    if args.output is None:
        pdf_name = Path(args.pdf_path).stem
        args.output = f"{pdf_name}_chapters"
    
    extract_chapters(args.pdf_path, args.output)


if __name__ == '__main__':
    main()
