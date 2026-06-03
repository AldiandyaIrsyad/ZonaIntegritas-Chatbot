"""Tests for thumbnail generation strategies."""

import pytest
from unittest.mock import patch, MagicMock

from app.infra.thumbnail import (
    ThumbnailContext,
    DefaultThumbnailStrategy,
    ImageThumbnailStrategy,
    PDFThumbnailStrategy
)

@pytest.fixture
def context():
    """Fixture providing a ThumbnailContext instance."""
    return ThumbnailContext()

def test_context_routing_pdf(context):
    """Test that context routes .pdf to PDFThumbnailStrategy."""
    strategy = context._strategies.get(".pdf")
    assert isinstance(strategy, PDFThumbnailStrategy)

def test_context_routing_image(context):
    """Test that context routes image extensions to ImageThumbnailStrategy."""
    for ext in [".jpg", ".jpeg", ".png"]:
        strategy = context._strategies.get(ext)
        assert isinstance(strategy, ImageThumbnailStrategy)

def test_context_fallback(context):
    """Test that context falls back to DefaultThumbnailStrategy for unknown extensions."""
    # It delegates in generate()
    result = context.generate("test.txt")
    assert result is None

def test_default_strategy():
    """Test the default strategy always returns None."""
    strategy = DefaultThumbnailStrategy()
    assert strategy.generate("test.unknown") is None

@patch("app.infra.thumbnail.Image")
def test_image_strategy_success(mock_image):
    """Test successful image thumbnail generation."""
    mock_img_instance = MagicMock()
    mock_image.open.return_value.__enter__.return_value = mock_img_instance
    
    # We don't actually save, we let the mock handle the save call
    # which writes nothing to BytesIO, but base64 encoding empty bytes is ''
    
    strategy = ImageThumbnailStrategy()
    result = strategy.generate("test.jpg")
    
    mock_image.open.assert_called_once_with("test.jpg")
    mock_img_instance.thumbnail.assert_called_once_with((200, 200))
    mock_img_instance.save.assert_called_once()
    assert result is not None
    assert result.startswith("data:image/png;base64,")

@patch("app.infra.thumbnail.Image")
def test_image_strategy_exception(mock_image):
    """Test image thumbnail generation returning None on exception."""
    mock_image.open.side_effect = Exception("Pillow error")
    
    strategy = ImageThumbnailStrategy()
    result = strategy.generate("test.jpg")
    
    assert result is None

@patch("fitz.open")
def test_pdf_strategy_success(mock_fitz_open):
    """Test successful PDF thumbnail generation."""
    mock_doc = MagicMock()
    mock_fitz_open.return_value = mock_doc
    mock_doc.page_count = 1
    
    mock_page = MagicMock()
    mock_doc.load_page.return_value = mock_page
    
    mock_pix = MagicMock()
    mock_page.get_pixmap.return_value = mock_pix
    mock_pix.tobytes.return_value = b"fake_png_data"
    
    strategy = PDFThumbnailStrategy()
    
    # Needs to be able to import fitz internally inside generate()
    # We patch fitz globally before it's imported in the method
    import sys
    fitz_module = MagicMock()
    fitz_module.open = mock_fitz_open
    sys.modules["fitz"] = fitz_module
    
    result = strategy.generate("test.pdf")
    
    assert result is not None
    assert result.startswith("data:image/png;base64,")

def test_pdf_strategy_exception():
    """Test PDF thumbnail generation returning None on exception/missing module."""
    import sys
    # Ensure fitz is not mockable/available to simulate import error or other error
    if "fitz" in sys.modules:
        del sys.modules["fitz"]
        
    strategy = PDFThumbnailStrategy()
    result = strategy.generate("test.pdf")
    
    assert result is None
