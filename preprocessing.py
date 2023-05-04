import math
import typing
import os
import shutil
import svgelements
from svgelements import *
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--dataset_path", type=str)
parser.add_argument("-sf", "--save_folder", type=str)
parser.add_argument("-op", "--overflowed_path", type=str)
parser.add_argument("-pa", "--paths_amount", type=int, default=9)
parser.add_argument("-spp", "--segments_per_path", type=int, default=15)
parser.add_argument("-x", "--max_x", type=int, default=512)
parser.add_argument("-y", "--max_y", type=int, default=512)

args = parser.parse_args()

# Path(args.save_folder).mkdir(parents=True, exist_ok=True)
dataset_path = 'D:\\dloads\\svgicons\\svgicons'
pathlist = Path(dataset_path).rglob('*.svg')
overflowed = 'D:\\dloads\\svgicons\\svgicons\\overflowed'
processed = 'D:\\dloads\\svgicons\\svgicons\\processed'

Path(overflowed).mkdir(parents=True, exist_ok=True)
Path(processed).mkdir(parents=True, exist_ok=True)

paths_amount = args.paths_amount
segments_per_path = args.segments_per_path
maxX = args.max_x
maxY = args.max_y

svg_parser = svgelements.SVG()

element = svg_parser.parse("C:\\Users\\glebm\\Desktop\\fruits\\apple-organic.svg")


def getSize(element):
    return element.height, element.width


class Normalizer:

    def __init__(self, height, width, max_x, maxY):
        self.maxY = maxY
        self.maxX = max_x
        self.width = width
        self.height = height

    def to_list(self, seg):
        s = str(seg.d(smooth=False))
        splitted = s.split()
        name = str(splitted[0]).upper()
        segmentArgs = [name]
        for pair in splitted[1:]:
            x, y = [float(i) for i in pair.split(",")]
            # if seg.relative and seg.start is not None:
            #     x += seg.start.x
            #     y += seg.start.y
            segmentArgs += self.normalized(x, y)
        return segmentArgs

    def normalized(self, x, y):
        x *= (self.maxX / self.width)
        y *= (self.maxY / self.height)
        return [x, y]


def line_curve(piece):
    startX, startY = piece.start.x, piece.start.y
    endX, endY = piece.end.x, piece.end.y
    middleX, middleY = (startX + endX) / 2, (startY + endY) / 2
    return CubicBezier(Point(startX, startY), Point(middleX, middleY), Point(middleX, middleY), Point(endX, endY))


def to_path(elem) -> svgelements.Path:
    if isinstance(elem, svgelements.Path):
        return elem
    else:
        shape = typing.cast(Shape, t)
        d = shape.d()
        fill = shape.fill
        new_path = svgelements.Path(d)
        new_path.fill = fill
        return new_path


def to_svg(paths, normalizer):
    svg_result = SVG()
    for path in paths:
        path_d = path["path"]
        fill = path["fill"]
        d = " ".join([" ".join([k if isinstance(k, str) else str(math.ceil(float(k))) for k in normalizer.to_list(e)])
                      for e in path_d])
        path_result = svgelements.Path(d)
        path_result.fill = fill
        path_result.stroke_width = (maxX // 50)
        path_result.stroke = Color("black")
        svg_result.append(path_result)
    svg_result.width = maxX
    svg_result.height = maxY
    return svg_result


def de_casteljau(res_paths):
    new_paths = []
    for p in res_paths:
        path = p["path"]
        if len(path) < 2:
            lkdsjvs = 928
        while len(path) < segments_per_path:
            max_ind = 1
            max_path_length = -1
            for i, seg in enumerate(path):
                if i == 0:
                    continue
                l = seg.length()
                if l > max_path_length:
                    max_path_length = l
                    max_ind = i
            segment = typing.cast(CubicBezier, path[max_ind])
            p0, p1, p2, p3 = segment.start, segment.control1, segment.control2, segment.end

            def middle(a, b):
                return Point((a.x + b.x) / 2, (a.y + b.y) / 2)

            m0, m1, m2 = middle(p0, p1), middle(p1, p2), middle(p2, p3)
            q0, q1 = middle(m0, m1), middle(m1, m2)
            b = middle(q0, q1)
            left = CubicBezier(p0, m0, q0, b)
            right = CubicBezier(b, q1, m2, p3)
            path[max_ind] = left
            path = path[:max_ind + 1] + [right] + path[max_ind + 1:]
        p["path"] = path


cnt = 0
fine = 0
for path in pathlist:
    if os.path.isfile(processed + "\\" + path.name) or os.path.isfile(overflowed + "\\" + path.name):
        continue
    try:
        element = typing.cast(SVG, svg_parser.parse(str(path)))
        if (path.name == "0xbtc.svg"):
            kldj = 1
        res_paths = []
        height, width = getSize(element)
        overflow = False
        normalizer = Normalizer(height, width, maxX, maxY)
        for t in element.elements(lambda e: isinstance(e, Shape)):
            svgPath = to_path(t)
            if (path.name == "0xbtc.svg"):
                print(svgPath.string_xml())
            start = svgPath.first_point
            pieces = []
            for i, piece in enumerate(svgPath):
                # print(type(piece))
                if isinstance(piece, Move):
                    piece = typing.cast(Move, piece)
                    if len(pieces) != 0:
                        fail = True
                        continue
                        res_paths.append({
                            "path": pieces,
                            "fill": svgPath.fill
                        })
                        pieces = []
                    pieces.append(piece)
                elif isinstance(piece, CubicBezier):
                    piece = typing.cast(CubicBezier, piece)
                    pieces.append(piece)
                elif isinstance(piece, Line):
                    piece = typing.cast(Line, piece)
                    pieces.append(line_curve(piece))
                elif isinstance(piece, Arc):
                    piece = typing.cast(Arc, piece)
                    res = [a for a in piece.as_cubic_curves()]
                    pieces += res
                elif isinstance(piece, Close):
                    pieces.append(line_curve(piece))
                elif isinstance(piece, QuadraticBezier):
                    piece = typing.cast(QuadraticBezier, piece)
                    # CP1 = QP0 + 2/3 *(QP1-QP0)
                    # CP2 = QP2 + 2/3 *(QP1-QP2)
                    qp0x, qp1x, qp2x = piece.start.x, piece.control.x, piece.end.x
                    qp0y, qp1y, qp2y = piece.start.y, piece.control.y, piece.end.y
                    pieces.append(CubicBezier(piece.start,
                                              Point(qp0x + 2 / 3 * (qp1x - qp0x), qp0y + 2 / 3 * (qp1y - qp0y)),
                                              Point(qp2x + 2 / 3 * (qp1x - qp2x), qp2y + 2 / 3 * (qp1y - qp2y)),
                                              piece.end))
                else:
                    print("unexpected: " + str(type(piece)))
            if fail:
                continue
            if len(pieces) > segments_per_path:
                overflow = True
                continue
            res_paths.append({
                "path": pieces,
                "fill": svgPath.fill
            })
        if fail:
            fail = False
            continue
        de_casteljau(res_paths)
        if len(res_paths) > paths_amount:
            overflow = True
        if overflow:
            print("Overflow: " + str(path))
            place = str(Path(overflowed + "\\" + str(path).replace(dataset_path, "").replace("\\", "_")))
            to_svg(res_paths, normalizer).write_xml(place)
            cnt += 1
        else:
            place = str(Path(processed + "\\" + str(path).replace(dataset_path, "").replace("\\", "_")))
            to_svg(res_paths, normalizer).write_xml(place)
            # print([" ".join(
            #     [" ".join([i if isinstance(i, str) else str(math.ceil(float(i))) for i in e]) for e in seg["path"]]) for seg
            #     in res_paths])
        cht = 9
    except:
        print("Exception on " + path.name)
print(cnt)
