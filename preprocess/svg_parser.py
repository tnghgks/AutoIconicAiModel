# preprocess/svg_parser.py

import re
from xml.etree import ElementTree as ET

TOKEN_RE = re.compile(
    r'[MmLlHhVvCcSsQqTtAaZz]'
    r'|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
)

def path_to_tokens(d: str) -> list[str]:
    return ["PATH_START", *TOKEN_RE.findall(d), "PATH_END"]

def svg_to_path_list(svg_text: str) -> list[str]:
    root = ET.fromstring(svg_text)
    paths = []
    ns_strip = lambda t: t.split('}')[-1]

    for elem in root.iter():
        tag = ns_strip(elem.tag)
        attr = elem.attrib
        d = None

        if tag == "path":
            d = attr.get("d", "")
        elif tag == "line":
            d = f"M {attr['x1']} {attr['y1']} L {attr['x2']} {attr['y2']}"
        elif tag in ("polygon", "polyline"):
            pts = attr.get("points", "")
            closed = tag == "polygon"
            coords = pts.replace(",", " ").split()
            if len(coords) < 4: continue
            segs = [f"L {coords[i]} {coords[i+1]}" for i in range(2, len(coords), 2)]
            d = f"M {coords[0]} {coords[1]} " + " ".join(segs)
            if closed:
                d += " Z"
        elif tag == "circle":
            cx, cy, r = float(attr.get("cx", 0)), float(attr.get("cy", 0)), float(attr.get("r", 0))
            if r == 0: continue
            d = f"M {cx + r} {cy} A {r} {r} 0 1 0 {cx - r} {cy} A {r} {r} 0 1 0 {cx + r} {cy}"
        elif tag == "rect":
            x, y = float(attr.get("x", 0)), float(attr.get("y", 0))
            w, h = float(attr.get("width", 0)), float(attr.get("height", 0))
            if w == 0 or h == 0: continue
            d = f"M {x} {y} L {x+w} {y} L {x+w} {y+h} L {x} {y+h} Z"
        elif tag == "ellipse":
            cx, cy = float(attr.get("cx", 0)), float(attr.get("cy", 0))
            rx, ry = float(attr.get("rx", 0)), float(attr.get("ry", 0))
            if rx == 0 or ry == 0: continue
            d = f"M {cx+rx} {cy} A {rx} {ry} 0 1 0 {cx-rx} {cy} A {rx} {ry} 0 1 0 {cx+rx} {cy}"
        
        if d:
            paths.append(d)

    return paths
