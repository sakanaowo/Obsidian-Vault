#!/usr/bin/env python3
"""
Simple PDF Splitter
Chia PDF thành các file nhỏ theo số trang

USAGE:
    python simple_pdf_split.py book.pdf
    python simple_pdf_split.py book.pdf -p 20
    python simple_pdf_split.py book.pdf -p 10 -o output_folder
"""
import fitz
import argparse
from pathlib import Path


def split_pdf(pdf_path, pages_per_file=10, output_dir=None):
    """
    Split PDF thành các file nhỏ

    Args:
        pdf_path: Đường dẫn PDF
        pages_per_file: Số trang mỗi file
        output_dir: Thư mục output (optional)
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    # Determine output directory
    if output_dir is None:
        output_dir = pdf_path.stem + "_split"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"PDF SPLITTER")
    print(f"{'='*60}")
    print(f"📄 PDF: {pdf_path.name}")
    print(f"📊 Total pages: {total_pages}")
    print(f"📑 Pages per file: {pages_per_file}")
    print(f"📁 Output: {output_dir}")
    print(f"{'='*60}\n")

    file_count = 0

    for i in range(0, total_pages, pages_per_file):
        end = min(i + pages_per_file, total_pages)

        # Create new PDF
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=i, to_page=end - 1)

        # Save with numbered filename
        filename = f"pages_{i+1:04d}-{end:04d}.pdf"
        filepath = output_dir / filename
        new_doc.save(str(filepath))
        new_doc.close()

        file_count += 1
        print(f"  ✓ {filename} ({end - i} pages)")

    doc.close()

    print(f"\n{'='*60}")
    print(f"✅ COMPLETED")
    print(f"{'='*60}")
    print(f"📊 Created {file_count} files")
    print(f"📁 Output: {output_dir}/")
    print(f"{'='*60}\n")

    return {
        "output_dir": str(output_dir),
        "file_count": file_count,
        "total_pages": total_pages,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Split PDF into smaller files by page count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s book.pdf
    → Split into 10-page files (default)
  
  %(prog)s book.pdf -p 20
    → Split into 20-page files
  
  %(prog)s book.pdf -p 10 -o my_output
    → Custom output directory
  
  %(prog)s book.pdf -p 5
    → Split into 5-page files (good for mobile reading)

Output:
  book_split/
  ├── pages_0001-0010.pdf
  ├── pages_0011-0020.pdf
  ├── pages_0021-0030.pdf
  └── ...
        """,
    )

    parser.add_argument("pdf", help="Input PDF file path")

    parser.add_argument(
        "-p",
        "--pages-per-file",
        type=int,
        default=10,
        help="Number of pages per output file (default: 10)",
    )

    parser.add_argument(
        "-o", "--output", help="Output directory (default: <pdf_name>_split)"
    )

    args = parser.parse_args()

    try:
        result = split_pdf(
            args.pdf, pages_per_file=args.pages_per_file, output_dir=args.output
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
