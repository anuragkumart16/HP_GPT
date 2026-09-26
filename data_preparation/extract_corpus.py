from pathlib import Path

import pymupdf

PDF_PATH = "data/harrypotter.pdf"
OUTPUT_PATH = "data/raw/corpus.txt"

pdf = pymupdf.open(PDF_PATH)

pages = []

for page_number, page in enumerate(pdf, start=1):
    text = page.get_text()

    if text.strip():
        pages.append(text)

    if page_number % 500 == 0:
        print(f"Processed {page_number}/{len(pdf)} pages")

corpus = "\n".join(pages)

Path(OUTPUT_PATH).write_text(corpus, encoding="utf-8")

print(f"\nFinished.")
print(f"Pages: {len(pdf)}")
print(f"Characters: {len(corpus):,}")
print(f"Output: {OUTPUT_PATH}")