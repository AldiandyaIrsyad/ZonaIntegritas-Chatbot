import asyncio
import os
import sys
from dotenv import load_dotenv
from pprint import pprint

from app.kb.infra.unstructured_client import UnstructuredClient
from app.thesis.chunking.logic import create_parent_chunks, split_into_children

load_dotenv()

async def main():
    pdf_samples = [
        "table.pdf",
        "SOP1.pdf",
        "SOP2.pdf",
        "informal.pdf",
        "presentation.pdf",
        "formal.pdf",
    ]
    
    base_dir = "/home/aldiandyath/repo/skripsi_app/experiments/pdf_processing/samples"
    
    # Initialize unstructured client
    unstructured_url = os.getenv("UNSTRUCTURED_API_URL", "http://localhost:8001")
    client = UnstructuredClient(base_url=unstructured_url, extract_images=True)
    
    for pdf_name in pdf_samples:
        pdf_path = os.path.join(base_dir, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            continue
            
        print(f"\n{'='*50}\nProcessing {pdf_name}\n{'='*50}")
        try:
            elements = await client.parse_pdf(pdf_path)
            print(f"Total elements: {len(elements)}")
            
            # Print first 10 elements to see how they are parsed
            print("\nFirst 10 elements:")
            for idx, el in enumerate(elements[:10]):
                text = el.text[:100].replace('\n', '\\n') + "..." if len(el.text) > 100 else el.text.replace('\n', '\\n')
                print(f"  [{idx}] {el.element_type}: {text}")
                if el.element_type == "Table" and "text_as_html" in el.metadata:
                    print(f"      HTML snippet: {el.metadata['text_as_html'][:100]}...")
                if el.element_type == "Image":
                    print(f"      Image Path: {el.metadata.get('image_path', 'None')}")
                    
            parent_chunks = create_parent_chunks(elements, "dummy_doc_id")
            print(f"\nTotal Parent Chunks: {len(parent_chunks)}")
            for idx, pc in enumerate(parent_chunks[:3]):
                print(f"  Parent [{idx}] Type: {pc.content_type.value}, Breadcrumbs: {pc.breadcrumbs}")
                print(f"      Text: {pc.text[:100].replace(chr(10), ' ')}...")
                
        except Exception as e:
            print(f"Failed to process {pdf_name}: {e}")
            
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
