"""
Build the system-overview figure of the manuscript (draw.io): the CBGT circuit as simulated, with
its inputs, outputs and connection types, and spiking model output in place of a continuous beta trace.

Starts from the unmodified draw.io source and writes a new file; panels C and D are untouched.
  Panel A  replaced by the CBGT circuit as implemented in simulate_network_optimized.py: eight
           populations; excitatory (arrow) and inhibitory (filled circle) connections read from the synaptic
           currents; DBS current into STN; GPi as the biomarker site; all 80 neurons observed.
  Panel B  labels added: observation, reward r(beta, E), actions, DBS parameter ranges; the generic
           network-cartoon icon of the model is replaced by a thumbnail raster of real model output
           (data from scripts/run_rev_activity.py, PD; illustrative subset: 5 of 10 neurons per
           population, 300 ms).

The draw.io source of the original figure is not part of this repository; pass its path and the
output path:

  python scripts/make_rev_fig2_drawio.py system_overview_v6.drawio system_overview_v15.drawio
"""

import base64
import html
import io
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

STIM, SENSE, INK, INK2 = '#eb6834', '#2a78d6', '#0b0b0b', '#52514e'   # categorical palette slots 1-2 + text inks
FONT = 'fontFamily=Arial;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DArial;'
NODE = f'rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f3;strokeColor={INK2};fontSize=7;arcSize=20;{FONT}'
EDGE = f'edgeStyle=none;rounded=0;html=1;strokeColor={INK};strokeWidth=1;endSize=4;startSize=4;{FONT}'
EXC = 'endArrow=block;endFill=1;'
INH = 'endArrow=oval;endFill=1;endSize=5;'   # filled circle: distinct from node outlines
TEXT = f'text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;rounded=0;{FONT}'

W, H = 36, 14
NODES = {   # id: (label, x, y) top-left, absolute page coordinates (panel A spans 10..210)
    'CorE': ('Cortex E', 52, 46), 'CorI': ('Cortex I', 120, 46),
    'D1': ('Str-D1', 22, 82), 'D2': ('Str-D2', 92, 82),
    'GPi': ('GPi', 22, 118), 'GPe': ('GPe', 92, 118), 'STN': ('STN', 160, 118),
    'TH': ('TH', 22, 154),
}
# (source, target, kind, exit (x, y), entry (x, y), waypoints); from the simulator's synaptic currents
EDGES = [
    ('CorE', 'CorI', EXC, (1, .3), (0, .3), []),                 # Iei: E -> I
    ('CorI', 'CorE', INH, (0, .7), (1, .7), []),                 # Iie: I -> E
    ('CorE', 'D1', EXC, (.25, 1), (.5, 0), []),                  # Icorstr6
    ('CorE', 'D2', EXC, (.75, 1), (.5, 0), []),                  # Icorstr5
    ('CorE', 'STN', EXC, (.5, 0), (.35, 0), [(70, 40), (172.6, 40)]),   # Icorsn AMPA/NMDA (hyperdirect)
    ('D1', 'D1', INH, (1, .25), (1, .75), [(65, 85.5), (65, 92.5)]),    # Igaba6: S8 is driven by V6 (D1 -> D1)
    ('D2', 'D2', INH, (1, .25), (1, .75), [(135, 85.5), (135, 92.5)]),  # Igaba5: S1c is driven by V5 (D2 -> D2)
    ('D1', 'GPi', INH, (.5, 1), (.5, 0), []),                    # Istrgpi
    ('D2', 'GPe', INH, (.5, 1), (.5, 0), []),                    # Istrgpe
    ('GPe', 'STN', INH, (1, .3), (0, .3), []),                   # Igesn
    ('STN', 'GPe', EXC, (0, .7), (1, .7), []),                   # Isngeampa/nmda
    ('GPe', 'GPi', INH, (0, .5), (1, .5), []),                   # Igigi (GPe -> GPi)
    ('GPe', 'GPe', INH, (.25, 1), (.75, 1), [(101, 138), (119, 138)]),   # Igege (GPe -> GPe)
    ('STN', 'GPi', EXC, (.5, 1), (.75, 1), [(178, 143), (49, 143)]),     # Isngi
    ('GPi', 'TH', INH, (.5, 1), (.5, 0), []),                    # Igith
    ('TH', 'CorE', EXC, (0, .5), (0, .5), [(15, 161), (15, 53)]),        # Ithcor
]


RASTER_SRC = os.path.join(os.path.dirname(__file__), '..', 'data/revision/activity/seed42_pd.json')
KEYS = ('TH_APs', 'STN_APs', 'GPe_APs', 'GPi_APs', 'Striat_APs_indr', 'Striat_APs_dr', 'Cor_APs')


def raster_png_b64(path=RASTER_SRC, t0=1.0, win=0.3, frac=0.5, width_px=580, height_px=520):
    """Thumbnail spike raster as base64 PNG, aspect matched to the icon slot. Illustrative subset so
    spikes stay legible at icon size: every other neuron of each population (frac), 300 ms (win)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = json.load(open(path))
    fig = plt.figure(figsize=(width_px / 200, height_px / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    row = 0
    for k in KEYS:
        trains = run['spikes'][k]
        for i in np.linspace(0, len(trains) - 1, round(frac * len(trains))).round().astype(int):
            s = np.asarray(trains[i])
            s = s[(s >= t0) & (s < t0 + win)] - t0
            ax.vlines(s, row + 0.1, row + 0.9, lw=2.6, color=INK)
            row += 1
        ax.axhline(row, lw=0.8, color='#b3b2ad')
    ax.set_xlim(0, win); ax.set_ylim(row, 0); ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='white')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def cell(root, cid, value, style, x=None, y=None, w=None, h=None, vertex=True, src=None, tgt=None,
         points=None, source_point=None, target_point=None):
    c = ET.SubElement(root, 'mxCell', {'id': cid, 'value': value, 'style': style, 'parent': '1'})
    if vertex:
        c.set('vertex', '1')
        ET.SubElement(c, 'mxGeometry', {'x': str(x), 'y': str(y), 'width': str(w), 'height': str(h), 'as': 'geometry'})
    else:
        c.set('edge', '1')
        if src:
            c.set('source', src)
        if tgt:
            c.set('target', tgt)
        g = ET.SubElement(c, 'mxGeometry', {'relative': '1', 'as': 'geometry'})
        if source_point:
            ET.SubElement(g, 'mxPoint', {'x': str(source_point[0]), 'y': str(source_point[1]), 'as': 'sourcePoint'})
        if target_point:
            ET.SubElement(g, 'mxPoint', {'x': str(target_point[0]), 'y': str(target_point[1]), 'as': 'targetPoint'})
        if points:
            arr = ET.SubElement(g, 'Array', {'as': 'points'})
            for px, py in points:
                ET.SubElement(arr, 'mxPoint', {'x': str(px), 'y': str(py)})
    return c


def main(src, dst):
    tree = ET.parse(src)
    root = tree.getroot().find('.//root')
    cells = {c.get('id'): c for c in root.iter('mxCell')}

    # --- remove the old panel A group and everything inside it
    group_id = next(k for k in cells if k.endswith('JfQ-29'))
    def descendants(pid):
        out = []
        for c in cells.values():
            if c.get('parent') == pid:
                out += [c] + descendants(c.get('id'))
        return out
    old = descendants(group_id)
    container_style = max((c for c in old if 'rounded=1' in (c.get('style') or '')),
                          key=lambda c: float(c.find('mxGeometry').get('width', 0))).get('style')
    title_cell = next(c for c in old if 'A) Biomarker' in html.unescape(c.get('value') or ''))
    title_value = html.unescape(title_cell.get('value')).replace('A) Biomarker', 'A) CBGT circuit model')
    title_style = title_cell.get('style')
    for c in old + [cells[group_id]]:
        root.remove(c)

    # --- new panel A
    cell(root, 'revA-box', '', container_style, 10, 10, 200, 200)
    cell(root, 'revA-title', title_value, title_style, 10, 10, 150, 30)
    for nid, (lab, x, y) in NODES.items():
        style = NODE
        if nid == 'STN':
            style = NODE.replace(f'strokeColor={INK2}', f'strokeColor={STIM};strokeWidth=1.5')
        if nid == 'GPi':
            style = NODE.replace(f'strokeColor={INK2}', f'strokeColor={SENSE};strokeWidth=1.5')
        cell(root, f'revA-{nid}', lab, style, x, y, W, H)
    for i, (s, t, kind, ex, en, pts) in enumerate(EDGES):
        extra = ''
        st = EDGE + kind + extra + f'exitX={ex[0]};exitY={ex[1]};exitDx=0;exitDy=0;entryX={en[0]};entryY={en[1]};entryDx=0;entryDy=0;'
        cell(root, f'revA-e{i}', '', st, vertex=False, src=f'revA-{s}', tgt=f'revA-{t}', points=pts)
    # DBS input to STN
    cell(root, 'revA-dbs', f'<font color="{STIM}"><b>DBS</b></font>', TEXT + 'align=center;verticalAlign=middle;fontSize=7;', 174, 91, 26, 12)
    cell(root, 'revA-dbs-e', '', EDGE.replace(f'strokeColor={INK}', f'strokeColor={STIM}') + EXC + 'entryX=0.75;entryY=0;entryDx=0;entryDy=0;',
         vertex=False, tgt='revA-STN', source_point=(187, 103))   # vertical, clear of the hyperdirect input at 0.35
    # legend and annotations
    cell(root, 'revA-lg1', '', EDGE + EXC, vertex=False, source_point=(68, 161), target_point=(84, 161))   # right of TH
    cell(root, 'revA-lg1t', 'excitatory', TEXT + 'align=left;verticalAlign=middle;fontSize=7.5;', 87, 155, 45, 12)
    cell(root, 'revA-lg2', '', EDGE + INH, vertex=False, source_point=(132, 161), target_point=(148, 161))
    cell(root, 'revA-lg2t', 'inhibitory', TEXT + 'align=left;verticalAlign=middle;fontSize=7.5;', 151, 155, 45, 12)
    note = (f'<font color="{STIM}">DBS current into STN</font> &nbsp;&middot;&nbsp; '
            f'<font color="{SENSE}">&beta; biomarker from GPi</font><br>'
            f'Observation: spikes of all 80 neurons<br>(10 per population)')
    cell(root, 'revA-note', note, TEXT + 'align=left;verticalAlign=top;fontSize=7.5;', 18, 168, 188, 30)   # raised 7 to clear the panel border

    # --- panel B labels
    def set_value(suffix, value, extra_style=None):
        c = next(v for k, v in cells.items() if k.endswith(suffix))
        c.set('value', value)
        if extra_style:
            c.set('style', c.get('style') + extra_style)
    set_value('uxq-40', 'CBGT spikes')
    g = next(v for k, v in cells.items() if k.endswith('uxq-40')).find('mxGeometry')
    g.set('x', '214'); g.set('width', '100')   # one line, clear of the raster box (centre kept at 264)
    icon_group = next(k for k in cells if k.endswith('Bs4h-846'))
    icon = next(c for c in cells.values() if c.get('parent') == icon_group and 'image=data' in (c.get('style') or ''))
    st = icon.get('style')
    i = st.index('image=data:image/png,')
    j = st.find(';', i)
    icon.set('style', st[:i] + 'image=data:image/png,' + raster_png_b64() + (st[j:] if j != -1 else ''))
    set_value('uxq-68', 'Reward r(&beta;, E)')
    set_value('uxq-67', 'DBS parameters<br><font style="font-size: 6px;">f &le; 180 Hz, &delta; 60&ndash;400 &micro;s, A &le; 250 &micro;A</font>')
    cell(root, 'revB-obs', 'Observation:<br>80 spike trains', TEXT + 'align=left;verticalAlign=top;fontSize=6;', 229, 42, 50, 18)
    cell(root, 'revB-act', 'Actions:<br>&Delta;f, &Delta;&delta;, &Delta;A<br>&isin; {&minus;1, 0, +1}', TEXT + 'align=left;verticalAlign=top;fontSize=6;', 385, 160, 40, 26)

    ET.indent(tree, space='  ')
    tree.write(dst, encoding='utf-8', xml_declaration=False)
    print('wrote', dst)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
