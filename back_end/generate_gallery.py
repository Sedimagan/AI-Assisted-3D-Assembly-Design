"""
generate_gallery.py — regenerates Best_models_for_training/gallery.html, a
thumbnail-view browser for the training corpus. Categories are auto-discovered
(any top-level folder that isn't an infra/quarantine folder), so adding a new
category folder and re-running this script is all that's needed to include it.

Usage:
    cd back_end
    ../.venv/bin/python generate_gallery.py
"""

import os
import html as html_lib
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent / "Source_3d_models" / "Best_models_for_training"
EXCLUDE_DIRS = {"non_compatible_formats", "rejected", "_quarantine_stall"}
IMG_EXTS  = {"png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff"}
STEP_EXTS = {"stp", "step"}


def discover_categories() -> list:
    return sorted(
        d.name for d in ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name not in EXCLUDE_DIRS
    )


def urlquote(path: str) -> str:
    return quote(path)


def build() -> None:
    categories = discover_categories()
    sections_html, nav_html = [], []
    total_folders = total_images = total_no_image_folders = 0

    for cat in categories:
        catpath = ROOT / cat
        folders = sorted(
            d.name for d in catpath.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        cat_id = cat.replace(" ", "_")
        nav_html.append(
            f'<a href="#{cat_id}" class="navlink">{html_lib.escape(cat)} '
            f'<span class="navcount">{len(folders)}</span></a>'
        )

        cards = []
        for folder in folders:
            total_folders += 1
            full = catpath / folder
            images_dir = full / "Images"
            step_file = next(
                (f.name for f in sorted(full.iterdir())
                 if f.is_file() and f.suffix.lstrip(".").lower() in STEP_EXTS),
                None
            )

            imgs = []
            if images_dir.is_dir():
                imgs = sorted(
                    f.name for f in images_dir.iterdir()
                    if f.suffix.lstrip(".").lower() in IMG_EXTS
                )
            total_images += len(imgs)
            if not imgs:
                total_no_image_folders += 1

            thumbs = "".join(
                f'<a href="{urlquote(cat)}/{urlquote(folder)}/Images/{urlquote(im)}" '
                f'target="_blank" class="thumb-link">'
                f'<img class="thumb" src="{urlquote(cat)}/{urlquote(folder)}/Images/{urlquote(im)}" '
                f'loading="lazy" alt="{html_lib.escape(im)}"></a>'
                for im in imgs
            )
            if not thumbs:
                thumbs = '<div class="no-img">No images</div>'

            step_badge = (
                f'<span class="stepfile" title="{html_lib.escape(step_file)}">'
                f'{html_lib.escape(step_file)}</span>'
                if step_file else '<span class="stepfile missing">no step/stp file</span>'
            )

            cards.append(f'''
            <div class="card" data-name="{html_lib.escape(folder.lower())}">
              <div class="card-header">
                <span class="foldername">{html_lib.escape(folder)}</span>
                {step_badge}
                <span class="imgcount">{len(imgs)} img</span>
              </div>
              <div class="thumbs">{thumbs}</div>
            </div>''')

        sections_html.append(f'''
        <section id="{cat_id}">
          <h2>{html_lib.escape(cat)} <span class="section-count">({len(folders)} folders)</span></h2>
          <div class="grid">
            {"".join(cards)}
          </div>
        </section>''')

    html_doc = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Best_models_for_training — Image Gallery</title>
<style>
  :root {{
    --bg: #f5f7fa; --panel: #ffffff; --border: #dbe2ea; --text: #1c2733; --muted: #64748b;
    --accent: #3b82f6; --accent-bg: #eaf1fe;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #0f1520; --panel: #161f2e; --border: #2a3648; --text: #e6edf5; --muted: #8aa0b8; --accent: #5b9bd5; --accent-bg: #1b2a3f; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    background: var(--bg); color: var(--text);
  }}
  header {{
    position: sticky; top: 0; z-index: 10; background: var(--panel); border-bottom: 1px solid var(--border);
    padding: 14px 24px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }}
  header h1 {{ font-size: 1.15rem; margin: 0; white-space: nowrap; }}
  .summary {{ color: var(--muted); font-size: 0.85rem; }}
  #search {{
    flex: 1; min-width: 180px; max-width: 320px; padding: 7px 12px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 0.9rem;
  }}
  nav {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .navlink {{
    color: var(--accent); text-decoration: none; font-size: 0.85rem; font-weight: 600;
    padding: 4px 10px; border-radius: 6px; background: var(--accent-bg);
  }}
  .navcount {{ color: var(--muted); font-weight: 400; }}
  main {{ padding: 24px; max-width: 1400px; margin: 0 auto; }}
  section {{ margin-bottom: 40px; }}
  h2 {{ font-size: 1.3rem; margin: 0 0 14px; }}
  .section-count {{ color: var(--muted); font-weight: 400; font-size: 0.95rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }}
  .card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px;
    display: flex; flex-direction: column; gap: 8px;
  }}
  .card-header {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
  .foldername {{ font-weight: 700; font-size: 0.92rem; }}
  .stepfile {{
    font-size: 0.72rem; color: var(--accent); background: var(--accent-bg); border-radius: 4px;
    padding: 1px 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 160px;
  }}
  .stepfile.missing {{ color: #ef5a5a; background: rgba(239,90,90,0.1); }}
  .imgcount {{ margin-left: auto; font-size: 0.75rem; color: var(--muted); }}
  .thumbs {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .thumb-link {{ display: block; }}
  .thumb {{
    width: 72px; height: 72px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border);
    background: #fff;
  }}
  .no-img {{ color: var(--muted); font-size: 0.8rem; font-style: italic; padding: 6px 0; }}
  .card.hidden {{ display: none; }}
</style>
</head>
<body>
<header>
  <h1>📁 Best_models_for_training — Image Gallery</h1>
  <input type="text" id="search" placeholder="Filter by folder name..." oninput="filterCards()">
  <nav>{"".join(nav_html)}</nav>
  <span class="summary">{total_folders} folders · {total_images} images · {total_no_image_folders} folders with no images</span>
</header>
<main>
  {"".join(sections_html)}
</main>
<script>
function filterCards() {{
  const q = document.getElementById('search').value.trim().toLowerCase();
  document.querySelectorAll('.card').forEach(c => {{
    c.classList.toggle('hidden', q && !c.dataset.name.includes(q));
  }});
}}
</script>
</body>
</html>
'''

    out_path = ROOT / "gallery.html"
    out_path.write_text(html_doc)
    print(f"Categories: {', '.join(categories)}")
    print(f"Wrote {out_path}")
    print(f"Folders: {total_folders}, Images: {total_images}, Folders w/o images: {total_no_image_folders}")


if __name__ == "__main__":
    build()
