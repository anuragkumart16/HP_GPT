import pymupdf

pdf = pymupdf.open("Data/harrypotter.pdf")

print("Pages:", len(pdf))

for i, page in enumerate(pdf):
    text = page.get_text().strip()

    if len(text) > 500:
        print(f"\nFirst substantial page: {i + 1}")
        print("=" * 60)
        print(text[:2000])
        break