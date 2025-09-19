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


def convert_pdf_to_jpeg(
    input_pdf_path: str,
    dpi: int | None = 144,
    zoom: float | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
    jpg_quality: int = 95,
    jpg_subsampling: int = 0,
    jpg_progressive: bool = False,
    jpg_optimize: bool = False,
    colorspace_name: str = "rgb",
    output_format: str = "jpeg",
) -> None:
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

        # Determine default render scale (used if no max-size specified)
        if zoom is not None and zoom <= 0:
            raise ValueError("zoom must be > 0")
        if zoom is None:
            if dpi is None or dpi <= 0:
                raise ValueError("dpi must be > 0 if zoom not provided")
            default_zoom_factor = float(dpi) / 72.0
        else:
            default_zoom_factor = float(zoom)

        # Choose colorspace
        if colorspace_name.lower() == "rgb":
            colorspace = fitz.csRGB
        elif colorspace_name.lower() == "gray":
            colorspace = fitz.csGRAY
        elif colorspace_name.lower() == "cmyk":
            colorspace = fitz.csCMYK
        else:
            raise ValueError("colorspace must be one of: rgb, gray, cmyk")

        for page_index in range(page_count):
            page = document.load_page(page_index)
            # If max dimensions are provided, compute per-page scale to fit bounds; else use default
            if max_width is not None or max_height is not None:
                page_width_pts = float(page.rect.width)
                page_height_pts = float(page.rect.height)
                if page_width_pts <= 0 or page_height_pts <= 0:
                    raise ValueError("Invalid page dimensions")
                width_scale = (
                    float(max_width) / page_width_pts if max_width is not None else float("inf")
                )
                height_scale = (
                    float(max_height) / page_height_pts if max_height is not None else float("inf")
                )
                zoom_factor = min(width_scale, height_scale)
                if not (zoom_factor > 0 and zoom_factor != float("inf")):
                    raise ValueError("At least one of --max-width or --max-height must be > 0")
            else:
                zoom_factor = default_zoom_factor

            transform_matrix = fitz.Matrix(zoom_factor, zoom_factor)
            pixmap = page.get_pixmap(
                matrix=transform_matrix,
                alpha=False,
                colorspace=colorspace,
            )

            if output_format.lower() == "png":
                ext = "png"
            else:
                ext = "jpg"

            if page_count == 1:
                output_filename = f"{stem}.{ext}"
            else:
                output_filename = f"{stem}_p{page_index + 1}.{ext}"

            output_path = os.path.join(directory, output_filename)
            # Save with format-specific options
            if output_format.lower() == "png":
                pixmap.save(output_path)
            else:
                pixmap.save(
                    output_path,
                    jpg_quality=int(jpg_quality),
                    jpg_subsampling=int(jpg_subsampling),
                    jpg_progressive=bool(jpg_progressive),
                    jpg_optimize=bool(jpg_optimize),
                )
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

    zoom_group = parser.add_mutually_exclusive_group()
    zoom_group.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rendering DPI (default: 300). 72 DPI corresponds to 1.0 zoom.",
    )
    zoom_group.add_argument(
        "--zoom",
        type=float,
        default=None,
        help="Explicit zoom factor (overrides --dpi). Example: 2.0 ≈ 144 DPI.",
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Maximum output width in pixels. Preserves aspect ratio per page.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=None,
        help="Maximum output height in pixels. Preserves aspect ratio per page.",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (higher is better; default: 95)",
    )
    parser.add_argument(
        "--subsampling",
        choices=["444", "422", "420", "0", "1", "2"],
        default="444",
        help="Chroma subsampling: 444(0), 422(1), 420(2). Default: 444",
    )
    parser.add_argument(
        "--progressive",
        action="store_true",
        help="Write progressive JPEGs",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize JPEG Huffman tables",
    )
    parser.add_argument(
        "--colorspace",
        choices=["rgb", "gray", "cmyk"],
        default="rgb",
        help="Output colorspace (default: rgb)",
    )
    parser.add_argument(
        "--format",
        choices=["jpeg", "png"],
        default="jpeg",
        help="Output format (default: jpeg). PNG is lossless and sharper for text.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        subsampling_map: dict[str, int] = {"444": 0, "422": 1, "420": 2, "0": 0, "1": 1, "2": 2}
        convert_pdf_to_jpeg(
            input_pdf_path=args.pdf,
            dpi=args.dpi,
            zoom=args.zoom,
            max_width=args.max_width,
            max_height=args.max_height,
            jpg_quality=max(1, min(100, int(args.quality))),
            jpg_subsampling=subsampling_map[str(args.subsampling)],
            jpg_progressive=bool(args.progressive),
            jpg_optimize=bool(args.optimize),
            colorspace_name=args.colorspace,
            output_format=args.format,
        )
    except Exception as error:  # pragma: no cover
        sys.stderr.write(f"Error: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

