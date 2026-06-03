"""Tests for document parser infrastructure module."""

import os
import httpx
import pytest
import respx
import tempfile

from app.infra.document_parser import DocumentParser

@pytest.fixture
def parser():
    """Fixture providing a DocumentParser instance."""
    return DocumentParser(base_url="http://unstructured:8000")

@pytest.fixture
def sample_pdf():
    """Fixture providing a temporary fake PDF file."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF-1.4 fake content")
        tmp_path = tmp.name
        
    yield tmp_path
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

@respx.mock
async def test_parse_pdf_success(parser, sample_pdf):
    """Test successfully parsing a PDF file."""
    respx.post("http://unstructured:8000/general/v0/general").respond(
        json=[
            {"type": "Title", "text": "Test Document", "metadata": {"page_number": 1}},
            {"type": "NarrativeText", "text": "This is a test.", "metadata": {}},
            {"type": "Title", "text": " ", "metadata": {}},  # Empty text should be filtered
        ]
    )
    
    elements = await parser.parse_pdf(sample_pdf)
    
    assert len(elements) == 2
    assert elements[0].element_type == "Title"
    assert elements[0].text == "Test Document"
    assert elements[0].metadata == {"page_number": 1}
    
    assert elements[1].element_type == "NarrativeText"
    assert elements[1].text == "This is a test."
    assert elements[1].metadata == {}

async def test_parse_pdf_not_found(parser):
    """Test handling of a non-existent PDF file."""
    with pytest.raises(FileNotFoundError):
        await parser.parse_pdf("/path/to/nonexistent/file.pdf")

@respx.mock
async def test_parse_pdf_http_error(parser, sample_pdf):
    """Test handling of HTTP errors from the unstructured API."""
    respx.post("http://unstructured:8000/general/v0/general").respond(status_code=500)
    
    with pytest.raises(httpx.HTTPStatusError):
        await parser.parse_pdf(sample_pdf)

@respx.mock
async def test_parse_pdf_network_error(parser, sample_pdf):
    """Test handling of network exceptions."""
    respx.post("http://unstructured:8000/general/v0/general").mock(side_effect=httpx.ConnectError("Connection refused"))
    
    with pytest.raises(httpx.ConnectError):
        await parser.parse_pdf(sample_pdf)

async def test_close(parser):
    """Test closing the client."""
    await parser.close()
    assert parser._client.is_closed
