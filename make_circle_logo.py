# # corep_analysis/make_circle_logo.py

# """
# Convert a rectangular logo to a circular PNG with transparent background.
# Run: python make_circle_logo.py
# """

# from PIL import Image, ImageDraw, ImageOps

# INPUT = "assets/logo.png"          # your current logo
# OUTPUT = "assets/logo_circle.png"  # new circular version
# SIZE = 512                          # output size (square canvas)

# # Load
# img = Image.open(INPUT).convert("RGBA")

# # Crop to a square (center-crop)
# w, h = img.size
# side = min(w, h)
# left = (w - side) // 2
# top = (h - side) // 2
# img = img.crop((left, top, left + side, top + side))

# # Resize to target
# img = img.resize((SIZE, SIZE), Image.LANCZOS)

# # Create circular mask
# mask = Image.new("L", (SIZE, SIZE), 0)
# draw = ImageDraw.Draw(mask)
# draw.ellipse((0, 0, SIZE, SIZE), fill=255)

# # Apply mask
# output = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
# output.paste(img, (0, 0), mask=mask)

# # Save
# output.save(OUTPUT, "PNG")
# print(f"✅ Circular logo saved → {OUTPUT}")


"""
Convert a rectangular logo to a circular PNG with transparent background.
Automatically removes solid-color background.
Run: python make_circle_logo.py
"""

from PIL import Image, ImageDraw
import numpy as np

INPUT = "assets/logo.png"
OUTPUT = "assets/logo_circle.png"
SIZE = 512

# --- Load ---
img = Image.open(INPUT).convert("RGBA")
arr = np.array(img)

# --- Detect background color from the 4 corners ---
h, w = arr.shape[:2]
corners = [
    arr[0, 0],          # top-left
    arr[0, w - 1],      # top-right
    arr[h - 1, 0],      # bottom-left
    arr[h - 1, w - 1],  # bottom-right
]
# Average corner color = assumed background
bg_color = np.mean(corners, axis=0)[:3].astype(int)
print(f"🎨 Detected background color: RGB{tuple(bg_color)}")

# --- Make pixels similar to background transparent ---
tolerance = 30  # adjust if some background remains (increase) or logo gets cut (decrease)

r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
mask = (
    (np.abs(r.astype(int) - bg_color[0]) < tolerance) &
    (np.abs(g.astype(int) - bg_color[1]) < tolerance) &
    (np.abs(b.astype(int) - bg_color[2]) < tolerance)
)
arr[..., 3] = np.where(mask, 0, a)  # set alpha to 0 for background pixels

img = Image.fromarray(arr, "RGBA")

# --- Crop to square (center) ---
w, h = img.size
side = min(w, h)
left = (w - side) // 2
top = (h - side) // 2
img = img.crop((left, top, left + side, top + side))

# --- Resize ---
img = img.resize((SIZE, SIZE), Image.LANCZOS)

# --- Apply circular mask ---
circ_mask = Image.new("L", (SIZE, SIZE), 0)
draw = ImageDraw.Draw(circ_mask)
draw.ellipse((0, 0, SIZE, SIZE), fill=255)

output = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
output.paste(img, (0, 0), mask=circ_mask)

output.save(OUTPUT, "PNG")
print(f"✅ Circular logo saved → {OUTPUT}")


