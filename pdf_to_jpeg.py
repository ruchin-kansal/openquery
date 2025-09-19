#!/usr/bin/env python3

import argparse
import os
import sys

try:
    import fitz  # PyMuPDF
except Exception as import_error:  # pragma: no cover
    sys.stderr.write(
        "PyMuPDF (fitz) is required. Install with: pip install PyMuPDF\n"
    )
    raise


def convert_pdf_to_jpeg(input_pdf_path: str) -> None:
    """Convert a PDF into JPEG image(s).

    - If the PDF has one page, output is `<stem>.jpg`.
    - If multiple pages, outputs are `<stem>_p{n}.jpg` for each page (1-based).
    """

    if not os.path.isfile(input_pdf_path):
        raise FileNotFoundError(f"Input file not found: {input_pdf_path}")

    if not input_pdf_path.lower().endswith(".pdf"):
        raise ValueError("Input file must have a .pdf extension")

    directory = os.path.dirname(input_pdf_path) or "."
    stem = os.path.splitext(os.path.basename(input_pdf_path))[0]

    with fitz.open(input_pdf_path) as document:
        page_count = document.page_count
        if page_count == 0:
            raise ValueError("PDF contains no pages")

        # Use a zoom for better quality (approx 144 DPI with 2x zoom)
        zoom_factor = 2.0
        transform_matrix = fitz.Matrix(zoom_factor, zoom_factor)

        for page_index in range(page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=transform_matrix, alpha=False)

            if page_count == 1:
                output_filename = f"{stem}.jpg"
            else:
                output_filename = f"{stem}_p{page_index + 1}.jpg"

            output_path = os.path.join(directory, output_filename)
            pixmap.save(output_path)
            print(f"Wrote: {output_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PDF to JPEG using PyMuPDF (fitz)."
    )
    parser.add_argument(
        "pdf",
        metavar="PDF_PATH",
        help="Path to the input PDF file",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        convert_pdf_to_jpeg(args.pdf)
    except Exception as error:  # pragma: no cover
        sys.stderr.write(f"Error: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

