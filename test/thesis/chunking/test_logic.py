import pytest
from app.thesis.chunking.logic import create_parent_chunks, split_into_children
from app.thesis.chunking.models import ParsedElement, ParentChunkData, ContentType

def test_heading_stack_and_breadcrumbs():
    elements = [
        ParsedElement(element_type="Title", text="Chapter 1", metadata={"category_depth": 0}),
        ParsedElement(element_type="Text", text="Introduction to Chapter 1."),
        ParsedElement(element_type="Title", text="Section 1.1", metadata={"category_depth": 1}),
        ParsedElement(element_type="Text", text="This is the first section."),
        ParsedElement(element_type="Title", text="Article 1.1.1", metadata={"category_depth": 2}),
        ParsedElement(element_type="Text", text="Here is a deep article rule."),
        # Sibling section
        ParsedElement(element_type="Title", text="Section 1.2", metadata={"category_depth": 1}),
        ParsedElement(element_type="Text", text="This is the second section."),
    ]
    
    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)
    
    # Expecting chunks: 
    # 1. Chapter 1 (flushed when Section 1.1 title is found)
    # 2. Section 1.1 (flushed when Article 1.1.1 is found)
    # 3. Article 1.1.1 (flushed when Section 1.2 is found)
    # 4. Section 1.2 (flushed at end)
    assert len(parents) == 4
    
    # 1. Chapter 1
    assert parents[0].breadcrumbs == ["Chapter 1"]
    assert "[Context: Chapter 1]" in parents[0].text
    
    # 2. Section 1.1
    assert parents[1].breadcrumbs == ["Chapter 1", "Section 1.1"]
    assert "[Context: Chapter 1 > Section 1.1]" in parents[1].text
    
    # 3. Article 1.1.1
    assert parents[2].breadcrumbs == ["Chapter 1", "Section 1.1", "Article 1.1.1"]
    assert "[Context: Chapter 1 > Section 1.1 > Article 1.1.1]" in parents[2].text
    
    # 4. Section 1.2
    assert parents[3].breadcrumbs == ["Chapter 1", "Section 1.2"]
    assert "[Context: Chapter 1 > Section 1.2]" in parents[3].text

def test_split_into_children_retains_context():
    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text="[Context: Chapter 1 > Section 1.1]\n\nSentence one. Sentence two. Sentence three. Sentence four. Sentence five.",
        chunk_index=0,
        breadcrumbs=["Chapter 1", "Section 1.1"]
    )
    
    # Set max chars small enough to force a split
    children = split_into_children(parent, max_chars=80, overlap_chars=5)
    
    assert len(children) > 1
    for child in children:
        assert "[Context: Chapter 1 > Section 1.1]" in child.text
        assert child.breadcrumbs == ["Chapter 1", "Section 1.1"]


# ---------------------------------------------------------------------------
# Table protection tests
# ---------------------------------------------------------------------------

def test_table_becomes_own_parent_chunk():
    """A Table element should become its own parent chunk, not mixed with prose."""
    elements = [
        ParsedElement(element_type="Title", text="Section A", metadata={"category_depth": 0}),
        ParsedElement(element_type="NarrativeText", text="Some intro text."),
        ParsedElement(
            element_type="Table",
            text="<table><tr><td>Col1</td><td>Col2</td></tr></table>",
            metadata={"text_as_html": "<table>...</table>", "page_number": 1},
        ),
        ParsedElement(element_type="NarrativeText", text="Text after the table."),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    # Expect: [Section A text] [Table] [text after table]
    # The table is flushed as its own parent, separating it from prose
    table_parents = [p for p in parents if p.content_type == ContentType.TABLE]
    text_parents = [p for p in parents if p.content_type == ContentType.TEXT]

    assert len(table_parents) == 1, "Table should be its own parent chunk"
    assert "<table>" in table_parents[0].text
    assert table_parents[0].element_metadata.get("text_as_html") == "<table>...</table>"


def test_table_not_split_into_multiple_children():
    """A table parent chunk should produce a single child (no character splitting)."""
    table_html = "<table>" + "<tr><td>A</td><td>B</td></tr>" * 20 + "</table>"

    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text=f"[Context: Section A]\n\n{table_html}",
        chunk_index=0,
        breadcrumbs=["Section A"],
        content_type=ContentType.TABLE,
        element_metadata={"text_as_html": table_html},
    )

    # Even with a small max_chars, the table should NOT be split
    children = split_into_children(parent, max_chars=50, overlap_chars=5)

    assert len(children) == 1, "Table must not be character-split"
    assert "<table>" in children[0].text
    assert "</table>" in children[0].text
    assert children[0].content_type == ContentType.TABLE


def test_table_with_summary_produces_two_children():
    """A table with a summary in element_metadata should produce 2 children."""
    table_html = "<table><tr><td>A</td></tr></table>"
    summary = "This table contains column A with value A."

    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text=table_html,
        chunk_index=0,
        breadcrumbs=[],
        content_type=ContentType.TABLE,
        element_metadata={"text_as_html": table_html, "table_summary": summary},
    )

    children = split_into_children(parent, max_chars=512, overlap_chars=50)

    assert len(children) == 2
    # First child: full table
    assert children[0].text == table_html
    # Second child: summary
    assert summary in children[1].text


def test_table_preserves_breadcrumbs():
    """A table parent chunk should carry the breadcrumbs from its section."""
    elements = [
        ParsedElement(element_type="Title", text="Chapter 1", metadata={"category_depth": 0}),
        ParsedElement(
            element_type="Table",
            text="<table><tr><td>Data</td></tr></table>",
            metadata={"page_number": 5},
        ),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    table_parents = [p for p in parents if p.content_type == ContentType.TABLE]
    assert len(table_parents) == 1
    assert table_parents[0].breadcrumbs == ["Chapter 1"]
    assert "[Context: Chapter 1]" in table_parents[0].text


def test_consecutive_tables_become_separate_parents():
    """Two consecutive tables should become two separate parent chunks."""
    elements = [
        ParsedElement(
            element_type="Table",
            text="<table><tr><td>Table 1</td></tr></table>",
            metadata={},
        ),
        ParsedElement(
            element_type="Table",
            text="<table><tr><td>Table 2</td></tr></table>",
            metadata={},
        ),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    table_parents = [p for p in parents if p.content_type == ContentType.TABLE]
    assert len(table_parents) == 2
    assert "Table 1" in table_parents[0].text
    assert "Table 2" in table_parents[1].text


# ---------------------------------------------------------------------------
# Figure / VLM description tests
# ---------------------------------------------------------------------------

def test_figure_becomes_own_parent_chunk():
    """A Figure/Image element should become its own parent chunk."""
    elements = [
        ParsedElement(element_type="Title", text="Section A", metadata={"category_depth": 0}),
        ParsedElement(element_type="NarrativeText", text="Intro text."),
        ParsedElement(
            element_type="Image",
            text="Flowchart showing step 1 to step 5 with a decision point.",
            metadata={"image_path": "/tmp/page_1.png", "page_number": 2},
        ),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)

    figure_parents = [p for p in parents if p.content_type == ContentType.FIGURE]
    assert len(figure_parents) == 1
    assert "Flowchart" in figure_parents[0].text
    assert figure_parents[0].element_metadata.get("image_path") == "/tmp/page_1.png"


def test_figure_short_description_single_child():
    """A short VLM description should produce a single child (no splitting)."""
    description = "This flowchart shows the approval process with 3 steps."

    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text=description,
        chunk_index=0,
        breadcrumbs=[],
        content_type=ContentType.FIGURE,
    )

    children = split_into_children(parent, max_chars=512, overlap_chars=50)

    assert len(children) == 1
    assert children[0].text == description
    assert children[0].content_type == ContentType.FIGURE


def test_figure_long_description_splits_at_sentences():
    """A long VLM description should split at sentence boundaries."""
    # Build a description longer than max_chars
    description = "Step one is initiated. " * 50  # ~1000 chars

    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text=description,
        chunk_index=0,
        breadcrumbs=[],
        content_type=ContentType.FIGURE,
    )

    children = split_into_children(parent, max_chars=100, overlap_chars=10)

    assert len(children) > 1
    for child in children:
        assert child.content_type == ContentType.FIGURE


# ---------------------------------------------------------------------------
# Content type inheritance tests
# ---------------------------------------------------------------------------

def test_text_children_inherit_text_content_type():
    """Text parent chunks should produce TEXT children."""
    parent = ParentChunkData(
        id="p1",
        doc_id="d1",
        text="Sentence one. Sentence two. " * 50,
        chunk_index=0,
        breadcrumbs=[],
        content_type=ContentType.TEXT,
    )

    children = split_into_children(parent, max_chars=100, overlap_chars=10)

    assert len(children) > 1
    for child in children:
        assert child.content_type == ContentType.TEXT


def test_empty_elements_returns_empty_list():
    """Empty elements list should return empty parent list."""
    assert create_parent_chunks([], doc_id="doc1") == []


def test_empty_text_elements_skipped():
    """Elements with empty text should be skipped."""
    elements = [
        ParsedElement(element_type="NarrativeText", text=""),
        ParsedElement(element_type="Table", text=""),
    ]

    parents = create_parent_chunks(elements, doc_id="doc1", max_chars=1000)
    assert parents == []
