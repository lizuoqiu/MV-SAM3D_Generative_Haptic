#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def load_or_blank(path: Path | None, size: tuple[int, int], label: str) -> Image.Image:
    if path is None or not path.exists():
        img = Image.new("RGB", size, (30, 30, 30))
        d = ImageDraw.Draw(img)
        d.text((16, 16), f"{label}\nmissing", fill=(220, 220, 220))
        return img
    return Image.open(path).convert("RGB")


def fit_width(img: Image.Image, width: int) -> Image.Image:
    if img.width == width:
        return img
    h = int(round(img.height * (width / max(img.width, 1))))
    return img.resize((width, max(h, 1)))


def annotate(img: Image.Image, title: str) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, out.width, 34), fill=(0, 0, 0))
    d.text((10, 8), title, fill=(255, 255, 255))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 2x2 Task-2 verification summary image.")
    parser.add_argument("--rgb-image", required=True, help="Sample RGB image")
    parser.add_argument("--thermal-image", required=True, help="Sample thermal image (jpeg/png)")
    parser.add_argument("--verify-avg", required=True, help="Verification image for avg fusion")
    parser.add_argument("--verify-max", required=True, help="Verification image for max fusion")
    parser.add_argument("--output", required=True, help="Output summary png path")
    args = parser.parse_args()

    rgb = annotate(load_or_blank(Path(args.rgb_image), (960, 540), "RGB"), "Source RGB")
    thermal = annotate(load_or_blank(Path(args.thermal_image), (960, 540), "Thermal"), "Source Thermal")
    avg = annotate(load_or_blank(Path(args.verify_avg), (960, 540), "Verify AVG"), "Mapped Thermal (AVG)")
    maxv = annotate(load_or_blank(Path(args.verify_max), (960, 540), "Verify MAX"), "Mapped Thermal (MAX)")

    target_w = max(rgb.width, thermal.width, avg.width, maxv.width)
    rgb = fit_width(rgb, target_w)
    thermal = fit_width(thermal, target_w)
    avg = fit_width(avg, target_w)
    maxv = fit_width(maxv, target_w)

    cell_h = max(rgb.height, thermal.height, avg.height, maxv.height)

    def pad_h(im: Image.Image) -> Image.Image:
        if im.height == cell_h:
            return im
        out = Image.new("RGB", (im.width, cell_h), (20, 20, 20))
        out.paste(im, (0, 0))
        return out

    rgb = pad_h(rgb)
    thermal = pad_h(thermal)
    avg = pad_h(avg)
    maxv = pad_h(maxv)

    canvas = Image.new("RGB", (target_w * 2, cell_h * 2), (15, 15, 15))
    canvas.paste(rgb, (0, 0))
    canvas.paste(thermal, (target_w, 0))
    canvas.paste(avg, (0, cell_h))
    canvas.paste(maxv, (target_w, cell_h))

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"[summary] saved: {out}")


if __name__ == "__main__":
    main()
