"""Generate the Escriba app icon (.icns) using CoreGraphics."""

import subprocess
import tempfile
from pathlib import Path

import Quartz

PROJECT_DIR = Path(__file__).parent.parent.resolve()
RESOURCES_DIR = PROJECT_DIR / "resources"


def generate_icon_png(size: int, output_path: Path):
    """Draw the Escriba icon at the given size and save as PNG."""
    cs = Quartz.CGColorSpaceCreateDeviceRGB()
    ctx = Quartz.CGBitmapContextCreate(
        None, size, size, 8, size * 4, cs,
        Quartz.kCGImageAlphaPremultipliedLast,
    )

    # Background: rounded rect with gradient
    padding = size * 0.08
    rect = Quartz.CGRectMake(padding, padding, size - 2 * padding, size - 2 * padding)
    corner_radius = size * 0.22

    path = Quartz.CGPathCreateMutable()
    Quartz.CGPathAddRoundedRect(path, None, rect, corner_radius, corner_radius)
    Quartz.CGContextAddPath(ctx, path)
    Quartz.CGContextClip(ctx)

    # Gradient: dark indigo to deep purple
    gradient_colors = [
        0.12, 0.10, 0.22, 1.0,  # dark indigo
        0.25, 0.10, 0.38, 1.0,  # deep purple
    ]
    locations = [0.0, 1.0]
    gradient = Quartz.CGGradientCreateWithColorComponents(
        cs, gradient_colors, locations, 2,
    )
    Quartz.CGContextDrawLinearGradient(
        ctx, gradient,
        Quartz.CGPointMake(0, 0), Quartz.CGPointMake(size, size),
        0,
    )

    # Draw waveform bars (audio visualizer style)
    bar_count = 7
    bar_width = size * 0.055
    gap = size * 0.035
    total_w = bar_count * bar_width + (bar_count - 1) * gap
    start_x = (size - total_w) / 2
    center_y = size * 0.52

    heights = [0.12, 0.25, 0.40, 0.55, 0.40, 0.25, 0.12]

    for i, h_frac in enumerate(heights):
        x = start_x + i * (bar_width + gap)
        bar_h = size * h_frac
        y = center_y - bar_h / 2

        bar_rect = Quartz.CGRectMake(x, y, bar_width, bar_h)
        bar_path = Quartz.CGPathCreateMutable()
        bar_radius = bar_width / 2
        Quartz.CGPathAddRoundedRect(bar_path, None, bar_rect, bar_radius, bar_radius)

        Quartz.CGContextSetRGBFillColor(ctx, 1.0, 1.0, 1.0, 0.92)
        Quartz.CGContextAddPath(ctx, bar_path)
        Quartz.CGContextFillPath(ctx)

    # Recording dot (bottom right)
    dot_radius = size * 0.04
    dot_x = size * 0.72
    dot_y = size * 0.25
    Quartz.CGContextSetRGBFillColor(ctx, 1.0, 0.23, 0.19, 0.95)
    Quartz.CGContextFillEllipseInRect(
        ctx, Quartz.CGRectMake(
            dot_x - dot_radius, dot_y - dot_radius,
            dot_radius * 2, dot_radius * 2,
        ),
    )

    # Save as PNG
    image = Quartz.CGBitmapContextCreateImage(ctx)
    url = Quartz.CFURLCreateFromFileSystemRepresentation(
        None, str(output_path).encode(), len(str(output_path)), False,
    )
    dest = Quartz.CGImageDestinationCreateWithURL(url, "public.png", 1, None)
    Quartz.CGImageDestinationAddImage(dest, image, None)
    Quartz.CGImageDestinationFinalize(dest)


def create_icns(output_path: Path):
    """Generate a full .icns file with all required sizes."""
    RESOURCES_DIR.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        iconset = Path(tmpdir) / "Escriba.iconset"
        iconset.mkdir()

        sizes = [16, 32, 64, 128, 256, 512]
        for s in sizes:
            generate_icon_png(s, iconset / f"icon_{s}x{s}.png")
            generate_icon_png(s * 2, iconset / f"icon_{s}x{s}@2x.png")

        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(output_path)],
            check=True,
        )
    print(f"Icon created: {output_path}")


if __name__ == "__main__":
    create_icns(RESOURCES_DIR / "Escriba.icns")
