"""Bambu Studio thumbnail generation helpers."""

import io
import os
import zipfile
from typing import Any, Dict

from ..utils.logging import get_logger

logger = get_logger(__name__)


class BambuThumbnailWriter:
    """Writes Bambu Studio thumbnail assets into a 3MF ZIP."""

    THUMBNAIL_PATHS = (
        "Metadata/plate_1.png",
        "Metadata/plate_1_small.png",
        "Metadata/plate_no_light_1.png",
        "Metadata/top_1.png",
        "Metadata/pick_1.png",
    )

    def add_proper_thumbnails(
        self, zip_file: zipfile.ZipFile, optimization_results: Dict[str, Any]
    ) -> None:
        """Generate proper 512x512 PNG thumbnails for Bambu Studio."""
        try:
            from PIL import Image

            source_image_path = optimization_results.get("source_image_path")
            if source_image_path and os.path.exists(source_image_path):
                with Image.open(source_image_path) as img:
                    img = img.convert("RGBA")
                    img = img.resize((512, 512), Image.Resampling.LANCZOS)

                    self._add_image_thumbnail(zip_file, "Metadata/pick_1.png", img)
                    self._add_image_thumbnail(zip_file, "Metadata/plate_1.png", img)

                    small_img = img.resize((256, 256), Image.Resampling.LANCZOS)
                    small_img = small_img.resize((512, 512), Image.Resampling.NEAREST)
                    self._add_image_thumbnail(
                        zip_file, "Metadata/plate_1_small.png", small_img
                    )

                    dark_img = Image.new("RGBA", (512, 512), (40, 40, 40, 255))
                    dark_img.paste(img, (0, 0), img)
                    self._add_image_thumbnail(
                        zip_file, "Metadata/plate_no_light_1.png", dark_img
                    )
                    self._add_image_thumbnail(zip_file, "Metadata/top_1.png", img)
                    return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Could not generate proper thumbnails: {e}")

        self.add_placeholder_thumbnails(zip_file)

    def _add_image_thumbnail(
        self, zip_file: zipfile.ZipFile, path: str, image: Any
    ) -> None:
        """Add a PIL image as a thumbnail to the ZIP file."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        zip_file.writestr(path, buffer.getvalue())

    def add_placeholder_thumbnails(self, zip_file: zipfile.ZipFile) -> None:
        """Add proper sized placeholder PNG thumbnails."""
        try:
            from PIL import Image, ImageDraw

            placeholder = Image.new("RGBA", (512, 512), (128, 128, 128, 255))

            draw = ImageDraw.Draw(placeholder)
            for i in range(0, 512, 64):
                draw.line([(i, 0), (i, 512)], fill=(100, 100, 100, 255), width=1)
                draw.line([(0, i), (512, i)], fill=(100, 100, 100, 255), width=1)

            try:
                draw.text(
                    (256, 256), "BananaForge", fill=(255, 255, 255, 255), anchor="mm"
                )
            except Exception:
                pass

            buffer = io.BytesIO()
            placeholder.save(buffer, format="PNG", optimize=True)
            png_data = buffer.getvalue()

        except ImportError:
            png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"

        for path in self.THUMBNAIL_PATHS:
            zip_file.writestr(path, png_data)
