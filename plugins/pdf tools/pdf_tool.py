#!/usr/bin/env python3
"""
PDF Processing Tool - Comprehensive PDF Chapter & Image Extractor
Tách PDF thành chapters, chia nhỏ mỗi chapter thành các file 10 trang, và extract ảnh

QUICK USAGE:
    # Tự động xử lý toàn bộ: chapters + split 10 trang + extract images
    python pdf_tool.py book.pdf

    # Custom output folder
    python pdf_tool.py book.pdf -o my_output

    # Chỉ tách chapters (không split, không extract images)
    python pdf_tool.py book.pdf --chapters-only

    # Custom pages per split
    python pdf_tool.py book.pdf --pages-per-file 20

    # Không filter ảnh nhỏ
    python pdf_tool.py book.pdf --no-filter-images

FEATURES:
    ✅ Tách PDF theo chapters từ TOC
    ✅ Mỗi chapter được chia thành nhiều PDF nhỏ (10 trang/file)
    ✅ Extract ảnh từ mỗi chapter vào folder riêng
    ✅ Tổ chức folder cấu trúc rõ ràng
    ✅ Metadata JSON đầy đủ
    ✅ Tự động lọc icon/logo nhỏ

OUTPUT STRUCTURE:
    book.pdf → book_processed/
    ├── chapter_1/
    │   ├── pdfs/
    │   │   ├── chapter_1_pages_1-10.pdf
    │   │   ├── chapter_1_pages_11-20.pdf
    │   │   └── chapter_1_pages_21-30.pdf
    │   ├── images/
    │   │   ├── fig_1-1_example.png
    │   │   ├── fig_1-2_diagram.png
    │   │   └── images_metadata.json
    │   └── chapter_1_images.md
    ├── chapter_2/
    │   ├── pdfs/
    │   └── images/
    ├── processing_summary.json
    └── README.md

REQUIREMENTS:
    pip install PyMuPDF Pillow

For full help:
    python pdf_tool.py --help
"""

import fitz  # PyMuPDF
import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Import functions từ các modules có sẵn
try:
    from pdf_chapter_extractor import (
        get_toc_chapters,
        detect_chapters_by_pattern,
        sanitize_filename as sanitize_filename_chapter,
        extract_chapter_number,
        is_real_chapter
    )
except ImportError:
    print("⚠️  Warning: Could not import from pdf_chapter_extractor.py")
    print("    Some features may not work properly.")

try:
    from pdf_image_extractor import (
        pdf_image_extractor,
        find_image_caption,
        classify_image,
        is_figure_caption
    )
except ImportError:
    print("⚠️  Warning: Could not import from pdf_image_extractor.py")
    print("    Image extraction features may not work properly.")


def split_pdf_into_chunks(pdf_doc: fitz.Document, start_page: int, end_page: int, 
                          pages_per_file: int, output_dir: Path, 
                          chapter_num: int) -> List[Dict]:
    """
    Chia một chapter thành nhiều PDF nhỏ
    
    Args:
        pdf_doc: PyMuPDF document
        start_page: Trang bắt đầu chapter (1-indexed)
        end_page: Trang kết thúc chapter (1-indexed)
        pages_per_file: Số trang mỗi file
        output_dir: Thư mục output
        chapter_num: Số thứ tự chapter
    
    Returns:
        List of file info dicts
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_created = []
    total_pages = end_page - start_page + 1
    chunk_num = 1
    
    current_page = start_page
    
    while current_page <= end_page:
        chunk_end = min(current_page + pages_per_file - 1, end_page)
        
        # Tạo PDF chunk mới
        new_doc = fitz.open()
        new_doc.insert_pdf(pdf_doc, from_page=current_page - 1, to_page=chunk_end - 1)
        
        # Tên file: chapter_X_pages_Y-Z.pdf
        filename = f"chapter_{chapter_num}_pages_{current_page}-{chunk_end}.pdf"
        filepath = output_dir / filename
        
        new_doc.save(str(filepath))
        new_doc.close()
        
        files_created.append({
            'filename': filename,
            'path': str(filepath),
            'start_page': current_page,
            'end_page': chunk_end,
            'page_count': chunk_end - current_page + 1,
            'chunk_number': chunk_num
        })
        
        print(f"      → {filename}")
        
        current_page = chunk_end + 1
        chunk_num += 1
    
    return files_created


def process_pdf_comprehensive(
    pdf_path: str,
    output_dir: Optional[str] = None,
    pages_per_file: int = 10,
    chapters_only: bool = False,
    extract_images: bool = True,
    filter_images: bool = True,
    min_image_width: int = 200,
    min_image_height: int = 200,
    toc_level: int = 1,
    pattern: Optional[str] = None,
    include_all_sections: bool = False
) -> Dict:
    """
    Xử lý PDF toàn diện: chapters + split + extract images
    
    Args:
        pdf_path: Đường dẫn PDF
        output_dir: Thư mục output
        pages_per_file: Số trang mỗi file khi split
        chapters_only: Chỉ tách chapters, không split/extract
        extract_images: Có extract ảnh không
        filter_images: Có lọc icon/logo nhỏ không
        min_image_width: Chiều rộng tối thiểu của ảnh
        min_image_height: Chiều cao tối thiểu của ảnh
        toc_level: Level của TOC
        pattern: Pattern để detect chapters
        include_all_sections: Bao gồm tất cả sections
    
    Returns:
        Dict với kết quả xử lý
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Tạo output directory
    doc_name = pdf_path.stem
    short_name = sanitize_filename_chapter(doc_name, max_length=40)
    
    if output_dir is None:
        output_dir = pdf_path.parent / f"{short_name}_processed"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open PDF
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    
    print(f"\n{'='*80}")
    print(f"PDF COMPREHENSIVE PROCESSING TOOL")
    print(f"{'='*80}")
    print(f"📖 PDF: {pdf_path.name}")
    print(f"📄 Total pages: {total_pages}")
    print(f"📁 Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # STEP 1: Detect chapters
    print("🔍 STEP 1: Detecting chapters...")
    
    if pattern:
        print(f"   Using pattern: {pattern}")
        chapters = detect_chapters_by_pattern(doc, pattern)
    else:
        print(f"   Using TOC (level {toc_level})")
        chapters = get_toc_chapters(doc, toc_level, include_all_sections)
        
        if not chapters:
            print("   No chapters in TOC, trying pattern detection...")
            chapters = detect_chapters_by_pattern(doc, r"Chapter\s+\d+")
    
    if not chapters:
        print("❌ No chapters detected!")
        doc.close()
        return {'success': False, 'error': 'No chapters detected'}
    
    print(f"✅ Found {len(chapters)} chapters\n")
    
    # Print chapter summary
    print("📑 Chapter Summary:")
    print("-" * 80)
    for ch in chapters:
        print(f"   Chapter {ch['chapter_num']:2d}: {ch['title'][:50]:<50} (p.{ch['start_page']}-{ch['end_page']})")
    print("-" * 80 + "\n")
    
    # STEP 2: Process each chapter
    print("⚙️  STEP 2: Processing chapters...")
    
    all_results = []
    
    for idx, chapter in enumerate(chapters, 1):
        ch_num = chapter['chapter_num']
        ch_title = chapter['title']
        ch_start = chapter['start_page']
        ch_end = chapter['end_page']
        
        print(f"\n📘 [{idx}/{len(chapters)}] Chapter {ch_num}: {ch_title[:40]}")
        print(f"   Pages: {ch_start}-{ch_end} ({ch_end - ch_start + 1} pages)")
        
        # Tạo chapter folder
        chapter_folder = output_dir / f"chapter_{ch_num}"
        chapter_folder.mkdir(exist_ok=True)
        
        chapter_result = {
            'chapter_number': ch_num,
            'title': ch_title,
            'start_page': ch_start,
            'end_page': ch_end,
            'page_count': ch_end - ch_start + 1,
            'folder': str(chapter_folder),
            'pdf_chunks': [],
            'images': []
        }
        
        if chapters_only:
            # Chỉ extract chapter nguyên vẹn
            chapter_pdf_path = chapter_folder / f"chapter_{ch_num}_full.pdf"
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=ch_start - 1, to_page=ch_end - 1)
            new_doc.save(str(chapter_pdf_path))
            new_doc.close()
            
            chapter_result['full_pdf'] = str(chapter_pdf_path)
            print(f"   ✓ Saved full chapter: chapter_{ch_num}_full.pdf")
        else:
            # STEP 2a: Split chapter thành các file nhỏ
            print(f"   📄 Splitting into {pages_per_file}-page chunks...")
            pdfs_folder = chapter_folder / "pdfs"
            
            pdf_chunks = split_pdf_into_chunks(
                doc, ch_start, ch_end, pages_per_file, 
                pdfs_folder, ch_num
            )
            
            chapter_result['pdf_chunks'] = pdf_chunks
            print(f"   ✓ Created {len(pdf_chunks)} PDF chunks")
        
        # STEP 2b: Extract images
        if extract_images:
            print(f"   🖼️  Extracting images...")
            images_folder = chapter_folder / "images"
            
            # Extract chapter as temporary PDF for image extraction
            temp_chapter_pdf = chapter_folder / f"_temp_chapter_{ch_num}.pdf"
            temp_doc = fitz.open()
            temp_doc.insert_pdf(doc, from_page=ch_start - 1, to_page=ch_end - 1)
            temp_doc.save(str(temp_chapter_pdf))
            temp_doc.close()
            
            try:
                # Extract images
                img_result = pdf_image_extractor(
                    pdf_path=str(temp_chapter_pdf),
                    output_folder=str(images_folder),
                    min_width=min_image_width,
                    min_height=min_image_height,
                    filter_icons=filter_images,
                    create_markdown=True
                )
                
                chapter_result['images'] = img_result.get('images', [])
                chapter_result['total_images'] = img_result.get('total_extracted', 0)
                chapter_result['images_folder'] = str(images_folder)
                
                print(f"   ✓ Extracted {img_result.get('total_extracted', 0)} images")
                
            except Exception as e:
                print(f"   ⚠️  Image extraction failed: {e}")
                chapter_result['image_error'] = str(e)
            finally:
                # Xóa temp file
                if temp_chapter_pdf.exists():
                    temp_chapter_pdf.unlink()
        
        all_results.append(chapter_result)
    
    doc.close()
    
    # STEP 3: Create summary files
    print(f"\n📝 STEP 3: Creating summary files...")
    
    # Processing summary JSON
    summary = {
        'source_pdf': str(pdf_path),
        'source_name': doc_name,
        'total_pages': total_pages,
        'total_chapters': len(chapters),
        'output_directory': str(output_dir),
        'processing_date': datetime.now().isoformat(),
        'settings': {
            'pages_per_file': pages_per_file,
            'chapters_only': chapters_only,
            'extract_images': extract_images,
            'filter_images': filter_images,
            'min_image_size': f"{min_image_width}x{min_image_height}"
        },
        'chapters': all_results
    }
    
    summary_path = output_dir / 'processing_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Saved: processing_summary.json")
    
    # README
    readme_content = generate_readme(summary)
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"   ✓ Saved: README.md")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"✅ PROCESSING COMPLETED")
    print(f"{'='*80}")
    print(f"📊 Summary:")
    print(f"   • Source: {pdf_path.name}")
    print(f"   • Chapters processed: {len(chapters)}")
    print(f"   • Total PDF chunks: {sum(len(ch.get('pdf_chunks', [])) for ch in all_results)}")
    if extract_images:
        total_images = sum(ch.get('total_images', 0) for ch in all_results)
        print(f"   • Total images extracted: {total_images}")
    print(f"   • Output directory: {output_dir}")
    print(f"\n📁 Structure:")
    print(f"   {output_dir.name}/")
    for ch in all_results:
        ch_num = ch['chapter_number']
        print(f"   ├── chapter_{ch_num}/")
        if not chapters_only:
            print(f"   │   ├── pdfs/ ({len(ch.get('pdf_chunks', []))} files)")
        if extract_images and ch.get('total_images', 0) > 0:
            print(f"   │   └── images/ ({ch.get('total_images', 0)} images)")
    print(f"   ├── processing_summary.json")
    print(f"   └── README.md")
    print(f"{'='*80}\n")
    
    return {
        'success': True,
        'output_dir': str(output_dir),
        'summary_file': str(summary_path),
        'readme_file': str(readme_path),
        'chapters_processed': len(chapters),
        'results': all_results
    }


def generate_readme(summary: Dict) -> str:
    """Generate README.md content"""
    content = f"""# PDF Processing Results

## Source Information
- **File**: {summary['source_name']}
- **Total Pages**: {summary['total_pages']}
- **Processing Date**: {summary['processing_date']}

## Processing Settings
- **Pages per file**: {summary['settings']['pages_per_file']}
- **Chapters only**: {summary['settings']['chapters_only']}
- **Extract images**: {summary['settings']['extract_images']}
- **Filter small images**: {summary['settings']['filter_images']}
- **Min image size**: {summary['settings']['min_image_size']}

## Chapters Overview

"""
    
    for ch in summary['chapters']:
        content += f"### Chapter {ch['chapter_number']}: {ch['title']}\n\n"
        content += f"- **Pages**: {ch['start_page']}-{ch['end_page']} ({ch['page_count']} pages)\n"
        content += f"- **Folder**: `{Path(ch['folder']).name}/`\n"
        
        if ch.get('pdf_chunks'):
            content += f"- **PDF chunks**: {len(ch['pdf_chunks'])} files\n"
        
        if ch.get('total_images', 0) > 0:
            content += f"- **Images extracted**: {ch['total_images']}\n"
        
        content += "\n"
    
    content += f"""
## Directory Structure

```
{Path(summary['output_directory']).name}/
"""
    
    for ch in summary['chapters']:
        ch_num = ch['chapter_number']
        content += f"├── chapter_{ch_num}/\n"
        
        if ch.get('pdf_chunks'):
            content += f"│   ├── pdfs/\n"
            for pdf in ch['pdf_chunks'][:2]:  # Show first 2 as examples
                content += f"│   │   ├── {pdf['filename']}\n"
            if len(ch['pdf_chunks']) > 2:
                content += f"│   │   └── ... ({len(ch['pdf_chunks']) - 2} more files)\n"
        
        if ch.get('total_images', 0) > 0:
            content += f"│   ├── images/\n"
            content += f"│   │   └── ... ({ch['total_images']} images)\n"
            content += f"│   └── chapter_{ch_num}_images.md\n"
    
    content += """├── processing_summary.json
└── README.md
```

## Usage

### PDF Chunks
Each chapter is split into smaller PDFs for easier handling:
- Located in `chapter_X/pdfs/`
- Named as `chapter_X_pages_Y-Z.pdf`

### Images
Images extracted from each chapter:
- Located in `chapter_X/images/`
- Filtered to remove icons and small decorative images
- Includes captions and figure numbers when available
- Markdown index file for easy reference

### Metadata
- `processing_summary.json` - Complete processing details
- `images_metadata.json` - Image extraction metadata (in each chapter's images folder)

---
Generated by PDF Comprehensive Processing Tool
"""
    
    return content


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive PDF processing: chapters + split + image extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s book.pdf
    → Auto-process: extract chapters, split into 10-page chunks, extract images

  %(prog)s book.pdf -o my_output --pages-per-file 20
    → Custom output folder and 20 pages per file

  %(prog)s book.pdf --chapters-only
    → Only extract chapters, don't split or extract images

  %(prog)s book.pdf --no-filter-images
    → Extract all images including small icons

  %(prog)s book.pdf --pattern "Chapter \\d+"
    → Use custom pattern for chapter detection

Output Structure:
  book_processed/
  ├── chapter_1/
  │   ├── pdfs/
  │   │   ├── chapter_1_pages_1-10.pdf
  │   │   └── chapter_1_pages_11-20.pdf
  │   └── images/
  │       └── ... (extracted images)
  ├── chapter_2/
  └── processing_summary.json
        """
    )
    
    parser.add_argument('pdf', help='Input PDF file path')
    
    parser.add_argument('-o', '--output', 
                        help='Output directory (default: <pdf_name>_processed)')
    
    parser.add_argument('-p', '--pages-per-file', type=int, default=10,
                        help='Number of pages per split file (default: 10)')
    
    parser.add_argument('--chapters-only', action='store_true',
                        help='Only extract chapters, do not split or extract images')
    
    parser.add_argument('--no-images', action='store_true',
                        help='Do not extract images')
    
    parser.add_argument('--no-filter-images', action='store_true',
                        help='Do not filter small icons/logos')
    
    parser.add_argument('--min-width', type=int, default=200,
                        help='Minimum image width in pixels (default: 200)')
    
    parser.add_argument('--min-height', type=int, default=200,
                        help='Minimum image height in pixels (default: 200)')
    
    parser.add_argument('--toc-level', type=int, default=1,
                        help='TOC level to extract (default: 1)')
    
    parser.add_argument('--pattern',
                        help='Custom regex pattern for chapter detection')
    
    parser.add_argument('--include-all', action='store_true',
                        help='Include all TOC sections, not just chapters')
    
    args = parser.parse_args()
    
    try:
        result = process_pdf_comprehensive(
            pdf_path=args.pdf,
            output_dir=args.output,
            pages_per_file=args.pages_per_file,
            chapters_only=args.chapters_only,
            extract_images=not args.no_images,
            filter_images=not args.no_filter_images,
            min_image_width=args.min_width,
            min_image_height=args.min_height,
            toc_level=args.toc_level,
            pattern=args.pattern,
            include_all_sections=args.include_all
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
