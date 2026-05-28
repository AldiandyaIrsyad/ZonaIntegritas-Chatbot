import abc
import os
import base64
from io import BytesIO
from PIL import Image
import fitz


class ThumbnailStrategy(abc.ABC):
    @abc.abstractmethod
    def generate_thumbnail(self, file_path: str) -> str | None:
        """Generates a thumbnail and returns it as a base64 encoded data URI, or None if failed."""
        pass

class PDFThumbnailStrategy(ThumbnailStrategy):
    def generate_thumbnail(self, file_path: str) -> str | None:
        try:
            doc = fitz.open(file_path)
            if not doc.page_count:
                return None
            page = doc.load_page(0)
            # 1.5x scale gives ~108 DPI — readable and compact
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_data = pix.tobytes("png")
            b64 = base64.b64encode(img_data).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        except Exception:
            return None

class ImageThumbnailStrategy(ThumbnailStrategy):
    def generate_thumbnail(self, file_path: str) -> str | None:
        try:
            with Image.open(file_path) as img:
                img.thumbnail((200, 200))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/png;base64,{b64}"
        except Exception:
            return None

class DefaultThumbnailStrategy(ThumbnailStrategy):
    def generate_thumbnail(self, file_path: str) -> str | None:
        return None

class ThumbnailContext:
    def __init__(self):
        self._strategies = {
            ".pdf": PDFThumbnailStrategy(),
            ".jpg": ImageThumbnailStrategy(),
            ".jpeg": ImageThumbnailStrategy(),
            ".png": ImageThumbnailStrategy(),
        }
        self._default_strategy = DefaultThumbnailStrategy()

    def generate_thumbnail(self, file_path: str) -> str | None:
        _, ext = os.path.splitext(file_path)
        strategy = self._strategies.get(ext.lower(), self._default_strategy)
        return strategy.generate_thumbnail(file_path)
