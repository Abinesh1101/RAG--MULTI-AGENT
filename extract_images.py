import fitz  # PyMuPDF
import os
from pathlib import Path

# Create images folder if it doesn't exist
os.makedirs("data/images", exist_ok=True)

# Define folders
pdf_folder = Path("data/pdfs")
image_folder = Path("data/images")

image_count = 0

for pdf_file in pdf_folder.glob("*.pdf"):
    print(f"\n📄 Processing {pdf_file.name}...")
    doc = fitz.open(pdf_file)

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        print(f"  Page {page_num+1}: found {len(images)} images")

        # Extract embedded raster images
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Skip small icons/logos if desired
            if base_image["width"] < 200 or base_image["height"] < 200:
                continue

            image_name = f"{pdf_file.stem}_page{page_num+1}_img{img_index+1}.{image_ext}"
            image_path = image_folder / image_name

            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            print(f"  ✅ Saved: {image_name}")
            image_count += 1

        # Render full page if no raster images found
        if len(images) == 0:
            pix = page.get_pixmap(dpi=200)
            image_name = f"{pdf_file.stem}_page{page_num+1}_render.png"
            image_path = image_folder / image_name
            pix.save(image_path)
            print(f"  🖼️ Rendered full page as image: {image_name}")
            image_count += 1

    doc.close()

print(f"\n🎉 Extraction complete! Total images extracted: {image_count}")
