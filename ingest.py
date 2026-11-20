from pypdf import PdfReader


def extract_pages(path):
    """Read a PDF and return a list of {page, text} dicts."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": i, "text": text})
    return pages


def chunk_pages(pages, chunk_size=800, overlap=150):
    """Split each page's text into overlapping chunks, remembering the page."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    chunks = []
    for p in pages:
        text = p["text"].strip()
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({"page": p["page"], "text": text[start:end]})
            start += chunk_size - overlap
    return chunks


if __name__ == "__main__":
    pages = extract_pages("sample.pdf")
    chunks = chunk_pages(pages)

    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"\n[page {c['page']}] {c['text']}")
        print("-" * 40)