"""
Rough geometry preview of a draw.io file (no draw.io install needed): boxes, text, embedded images
and edges, with group offsets applied. For checking layout and collisions only; open the file in
draw.io for the real rendering.

  python scripts/preview_drawio.py file.drawio out.png
"""

import base64
import html
import io
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402


def style_dict(s):
    d = {}
    for part in (s or '').split(';'):
        if '=' in part:
            k, v = part.split('=', 1)
            d[k] = v
        elif part:
            d[part] = True
    return d


def main(path, out):
    root = ET.parse(path).getroot()
    cells = {c.get('id'): c for c in root.iter('mxCell')}

    def geom(c):
        g = c.find('mxGeometry')
        if g is None:
            return None
        return [float(g.get(k, 0) or 0) for k in ('x', 'y', 'width', 'height')]

    def offset(c):
        x = y = 0.0
        p = cells.get(c.get('parent'))
        while p is not None and p.get('id') not in ('0', '1'):
            g = geom(p)
            if g and not p.get('edge'):
                x += g[0]; y += g[1]
            p = cells.get(p.get('parent'))
        return x, y

    def box(c):
        g = geom(c)
        ox, oy = offset(c)
        return g[0] + ox, g[1] + oy, g[2], g[3]

    fig, ax = plt.subplots(figsize=(9, 9))
    for c in cells.values():
        st = style_dict(c.get('style'))
        if c.get('vertex') != '1' or geom(c) is None or 'group' in st:
            continue
        x, y, w, h = box(c)
        img = st.get('image')
        if isinstance(img, str) and img.startswith('data:image'):
            try:
                data = base64.b64decode(img.split(',', 1)[1])
                ax.imshow(Image.open(io.BytesIO(data)), extent=(x, x + w, y + h, y), zorder=1)
            except Exception:
                ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec='m', lw=0.5))
        elif 'text' not in st and st.get('strokeColor') != 'none':
            fc = st.get('fillColor', 'none')
            fc = 'none' if fc in ('none', 'default') else fc
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=%g' % (min(w, h) * 0.1 if st.get('rounded') == '1' else 0),
                                        fc=fc, ec=st.get('strokeColor', '#000000') if st.get('strokeColor') not in (None, 'default') else '#000000',
                                        lw=float(st.get('strokeWidth', 1)) * 0.6, zorder=2))
        v = re.sub(r'<br\s*/?>', '\n', html.unescape(c.get('value') or ''))
        v = re.sub('<[^>]+>', '', v).strip()
        if v:
            ax.text(x + w / 2, y + h / 2, v, ha='center', va='center', fontsize=float(st.get('fontSize', 12)) * 0.62,
                    color=st.get('fontColor', '#000000') if st.get('fontColor') not in (None, 'default') else '#000000', zorder=4)
    for c in cells.values():
        if c.get('edge') != '1':
            continue
        st = style_dict(c.get('style'))
        g = c.find('mxGeometry')
        pts = []
        src, tgt = cells.get(c.get('source')), cells.get(c.get('target'))

        def anchor(cell, kx, ky):
            x, y, w, h = box(cell)
            fx = float(st.get(kx)) if st.get(kx) not in (None, True) else 0.5
            fy = float(st.get(ky)) if st.get(ky) not in (None, True) else 0.5
            return x + w * fx, y + h * fy
        sp = g.find("mxPoint[@as='sourcePoint']") if g is not None else None
        tp = g.find("mxPoint[@as='targetPoint']") if g is not None else None
        ox, oy = offset(c)
        if src is not None:
            pts.append(anchor(src, 'exitX', 'exitY'))
        elif sp is not None:
            pts.append((float(sp.get('x', 0)) + ox, float(sp.get('y', 0)) + oy))
        arr = g.find("Array[@as='points']") if g is not None else None
        if arr is not None:
            pts += [(float(p.get('x', 0)) + ox, float(p.get('y', 0)) + oy) for p in arr.findall('mxPoint')]
        if tgt is not None:
            pts.append(anchor(tgt, 'entryX', 'entryY'))
        elif tp is not None:
            pts.append((float(tp.get('x', 0)) + ox, float(tp.get('y', 0)) + oy))
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            col = st.get('strokeColor', '#000000')
            col = '#000000' if col in (True, 'default', None) else col
            ax.plot(xs, ys, color=col, lw=0.7, zorder=3)
            end = st.get('endArrow', 'classic')
            if end not in ('none',):
                mk = '|' if end == 'baseDash' else 'o' if end == 'oval' else '>'
                ax.plot(xs[-1], ys[-1], marker=mk, color=col, ms=3.5, zorder=3)
    ax.set_xlim(0, 440); ax.set_ylim(440, 0); ax.set_aspect('equal'); ax.axis('off')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    print('saved', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
