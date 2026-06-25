"""Integration tests for the IngestWorker with VLM enrichment.

Tests the VLM enrichment step in the ingestion pipeline using mocked
adapters (no real HTTP calls, no real database).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.kb.application.ingest_worker import IngestWorker
from app.thesis.chunking.models import ParsedElement, ContentType


@pytest.fixture
def mock_db() -> MagicMock:
    """Mock AsyncSession."""
    db = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


@pytest.fixture
def mock_parser() -> AsyncMock:
    """Mock IDocumentParser."""
    parser = AsyncMock()
    return parser


@pytest.fixture
def mock_embedder() -> AsyncMock:
    """Mock ITextEmbedder."""
    embedder = AsyncMock()
    embedder.embed_texts = AsyncMock(return_value=[])
    return embedder


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    """Mock IVectorStore."""
    return AsyncMock()


@pytest.fixture
def mock_kb_repo() -> AsyncMock:
    """Mock IKBRepository."""
    repo = AsyncMock()
    repo.create_ingestion_task = AsyncMock(return_value=MagicMock(id="task-1"))
    repo.update_ingestion_task = AsyncMock()
    repo.get_pdf_by_id = AsyncMock(return_value=MagicMock(
        id="doc-1",
        pdf_path="/fake/path.pdf",
        ingestion_status="pending",
    ))
    repo.save_parent_chunks = AsyncMock()
    return repo


@pytest.fixture
def mock_vlm_enricher() -> AsyncMock:
    """Mock IVLMEnricher."""
    enricher = AsyncMock()
    enricher.describe_image = AsyncMock(return_value="A flowchart showing 5 steps.")
    enricher.set_pdf_path = MagicMock()
    return enricher


@pytest.fixture
def worker(
    mock_db: MagicMock,
    mock_parser: AsyncMock,
    mock_embedder: AsyncMock,
    mock_vector_store: AsyncMock,
    mock_kb_repo: AsyncMock,
    mock_vlm_enricher: AsyncMock,
) -> IngestWorker:
    """IngestWorker with mocked dependencies and VLM enricher."""
    return IngestWorker(
        db=mock_db,
        document_parser=mock_parser,
        text_embedder=mock_embedder,
        vector_store=mock_vector_store,
        kb_repo=mock_kb_repo,
        ivm_service=None,
        vlm_enricher=mock_vlm_enricher,
        image_dir="/tmp/test_images",
    )


class TestVLMEnrichmentStep:
    """Tests for the _enrich_figures method."""

    @pytest.mark.asyncio
    async def test_enrich_figures_calls_vlm_for_empty_text_figures(
        self,
        worker: IngestWorker,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """FIGURE elements with empty text should be enriched by the VLM."""
        elements = [
            ParsedElement(
                element_type="Image",
                text="",
                metadata={"image_path": "/tmp/page1.png", "page_number": 1},
            ),
            ParsedElement(
                element_type="NarrativeText",
                text="Some narrative text.",
                metadata={},
            ),
        ]

        result = await worker._enrich_figures(elements, "/fake/path.pdf", "doc-1")

        # VLM should have been called for the figure
        mock_vlm_enricher.describe_image.assert_called_once_with("/tmp/page1.png")
        # The figure's text should now be the VLM description
        assert result[0].text == "A flowchart showing 5 steps."
        # The narrative text should be unchanged
        assert result[1].text == "Some narrative text."

    @pytest.mark.asyncio
    async def test_enrich_figures_skips_figures_with_existing_text(
        self,
        worker: IngestWorker,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """FIGURE elements that already have text should not be re-enriched."""
        elements = [
            ParsedElement(
                element_type="Image",
                text="Already has a description.",
                metadata={"image_path": "/tmp/page1.png"},
            ),
        ]

        result = await worker._enrich_figures(elements, "/fake/path.pdf", "doc-1")

        mock_vlm_enricher.describe_image.assert_not_called()
        assert result[0].text == "Already has a description."

    @pytest.mark.asyncio
    async def test_enrich_figures_no_vlm_filters_empty_figures(
        self,
        mock_db: MagicMock,
        mock_parser: AsyncMock,
        mock_embedder: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_kb_repo: AsyncMock,
    ) -> None:
        """Without a VLM enricher, empty-text figures should be filtered out."""
        worker = IngestWorker(
            db=mock_db,
            document_parser=mock_parser,
            text_embedder=mock_embedder,
            vector_store=mock_vector_store,
            kb_repo=mock_kb_repo,
            vlm_enricher=None,
        )

        elements = [
            ParsedElement(
                element_type="Image",
                text="",
                metadata={"page_number": 1},
            ),
            ParsedElement(
                element_type="NarrativeText",
                text="Keep this text.",
                metadata={},
            ),
        ]

        result = await worker._enrich_figures(elements, "/fake/path.pdf", "doc-1")

        # The empty-text figure should be filtered out
        assert len(result) == 1
        assert result[0].text == "Keep this text."

    @pytest.mark.asyncio
    async def test_enrich_figures_vlm_failure_keeps_element(
        self,
        mock_db: MagicMock,
        mock_parser: AsyncMock,
        mock_embedder: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_kb_repo: AsyncMock,
    ) -> None:
        """When VLM enrichment fails, the element should be kept (fail-closed)."""
        mock_vlm = AsyncMock()
        mock_vlm.describe_image = AsyncMock(side_effect=Exception("VLM API error"))
        mock_vlm.set_pdf_path = MagicMock()

        worker = IngestWorker(
            db=mock_db,
            document_parser=mock_parser,
            text_embedder=mock_embedder,
            vector_store=mock_vector_store,
            kb_repo=mock_kb_repo,
            vlm_enricher=mock_vlm,
        )

        elements = [
            ParsedElement(
                element_type="Image",
                text="",
                metadata={"image_path": "/tmp/page1.png", "page_number": 1},
            ),
            ParsedElement(
                element_type="NarrativeText",
                text="Narrative.",
                metadata={},
            ),
        ]

        result = await worker._enrich_figures(elements, "/fake/path.pdf", "doc-1")

        # VLM was called but failed — the figure still has empty text
        # so it should be filtered out (can't embed empty text)
        assert len(result) == 1
        assert result[0].text == "Narrative."

    @pytest.mark.asyncio
    async def test_enrich_figures_no_figure_elements(
        self,
        worker: IngestWorker,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """If there are no FIGURE elements, VLM should not be called."""
        elements = [
            ParsedElement(element_type="NarrativeText", text="Text 1."),
            ParsedElement(element_type="Title", text="Heading"),
        ]

        result = await worker._enrich_figures(elements, "/fake/path.pdf", "doc-1")

        mock_vlm_enricher.describe_image.assert_not_called()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_enrich_figures_empty_list(
        self,
        worker: IngestWorker,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """Empty elements list should return empty without calling VLM."""
        result = await worker._enrich_figures([], "/fake/path.pdf", "doc-1")
        assert result == []
        mock_vlm_enricher.describe_image.assert_not_called()

    @pytest.mark.asyncio
    async def test_enrich_figures_sets_pdf_path_for_fallback(
        self,
        worker: IngestWorker,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """The VLM enricher should receive the PDF path (for fallback mode)."""
        elements = [
            ParsedElement(
                element_type="Image",
                text="",
                metadata={"image_path": "/tmp/page1.png", "page_number": 1},
            ),
        ]

        await worker._enrich_figures(elements, "/custom/path.pdf", "doc-1")

        mock_vlm_enricher.set_pdf_path.assert_called_once_with("/custom/path.pdf")


class TestIngestWorkerConstruction:
    """Tests for IngestWorker constructor and configuration."""

    def test_worker_accepts_vlm_enricher(
        self,
        mock_db: MagicMock,
        mock_parser: AsyncMock,
        mock_embedder: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_kb_repo: AsyncMock,
        mock_vlm_enricher: AsyncMock,
    ) -> None:
        """IngestWorker should accept an optional vlm_enricher parameter."""
        worker = IngestWorker(
            db=mock_db,
            document_parser=mock_parser,
            text_embedder=mock_embedder,
            vector_store=mock_vector_store,
            kb_repo=mock_kb_repo,
            vlm_enricher=mock_vlm_enricher,
            image_dir="/custom/images",
        )

        assert worker.vlm_enricher is mock_vlm_enricher
        assert worker.image_dir == "/custom/images"

    def test_worker_works_without_vlm_enricher(
        self,
        mock_db: MagicMock,
        mock_parser: AsyncMock,
        mock_embedder: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_kb_repo: AsyncMock,
    ) -> None:
        """IngestWorker should work without a VLM enricher (None default)."""
        worker = IngestWorker(
            db=mock_db,
            document_parser=mock_parser,
            text_embedder=mock_embedder,
            vector_store=mock_vector_store,
            kb_repo=mock_kb_repo,
        )

        assert worker.vlm_enricher is None
