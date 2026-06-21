import pytest
from app.thesis.chunking.logic import create_parent_chunks, split_into_children
from app.thesis.chunking.models import ParsedElement, ParentChunkData

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
