#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Assemble AGSi ``agsiGeometryAreaFromLines`` units into section polygons with shared lines.

The output mirrors the ``line_geometry`` / ``volumes`` layout of the Evo ``geological-sections``
object: a vertex array of section coordinates (u, v), chunks (offset, count) of consecutive
vertices, and volumes whose boundary rings are lists of (chunk, reversed) parts.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

import evo.logging

from ..reader.errors import AGSIInvalidDataError, AGSIUnsupportedGeometryError
from ..reader.models import AgsiGeometryAreaFromLines, AgsiGeometryFromFile, AgsiModel, AgsiModelElement
from ..reader.package import AgsiPackage
from .alignment import SectionFrame, check_lines_only, section_frame_for_model

logger = evo.logging.getLogger("data_converters")

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def profile_to_section(profile: npt.ArrayLike, name: str = "Profile") -> FloatArray:
    """Convert an AGSi 2D profile (chainage, elevation) to section coordinates (u, v).

    ``u`` is the chainage, unchanged (see :class:`~.alignment.SectionFrame`). The result runs in
    the direction of increasing chainage (a profile drawn the other way is reversed) and repeated
    consecutive vertices are removed.

    :raise AGSIUnsupportedGeometryError: For 3D profiles, or when chainage does not strictly
        increase along the profile (vertical steps, overhangs or reversals).
    :raise AGSIInvalidDataError: For fewer than two distinct vertices or non-finite coordinates.
    """
    line = np.asarray(profile, dtype=np.float64)
    if line.ndim != 2 or line.shape[1] not in (2, 3):
        raise AGSIInvalidDataError(f"{name}: expected (chainage, elevation) vertices, got shape {line.shape}.")
    if line.shape[1] == 3:
        raise AGSIUnsupportedGeometryError(
            f"{name}: 3D profiles are not supported; section profiles must be 2D (chainage, elevation)."
        )
    if not np.all(np.isfinite(line)):
        raise AGSIInvalidDataError(f"{name}: non-finite coordinates.")
    uv = line + 0.0  # copy, and normalise -0.0 so identical lines have identical bytes
    if len(uv):
        uv = uv[np.concatenate([[True], np.any(uv[1:] != uv[:-1], axis=1)])]
    if len(uv) < 2:
        raise AGSIInvalidDataError(f"{name}: a profile needs at least two distinct vertices.")
    if uv[-1, 0] < uv[0, 0]:
        uv = uv[::-1].copy()
    if np.any(np.diff(uv[:, 0]) <= 0):
        raise AGSIUnsupportedGeometryError(
            f"{name}: chainage must strictly increase along the profile; vertical steps, overhangs and "
            "reversals are not supported."
        )
    return uv


@dataclass(frozen=True)
class SectionUnit:
    """One AGSi area: its top and/or bottom line in section coordinates (see :func:`profile_to_section`)."""

    key: str
    top: Optional[FloatArray] = None
    bottom: Optional[FloatArray] = None


@dataclass
class SectionVolume:
    """Closed polygon of a unit: (chunk index, reversed) parts in ring order.

    The ring runs along the top line in the direction of increasing u, down the right edge, back
    along the bottom line and up the left edge. Edges are omitted where top and bottom meet.
    """

    key: str
    parts: list[tuple[int, bool]]


@dataclass
class SectionLine:
    """A distinct top/bottom line; a line shared by several units is stored once.

    :param parts: Chunk indices that together make up the line, in order of increasing u.
    :param top_of: Keys of the units whose top is this line.
    :param bottom_of: Keys of the units whose bottom is this line.
    :param generated: ``True`` for horizontal lines generated from ``top_elevation`` /
        ``base_elevation`` for units without a top or bottom.
    """

    parts: list[int]
    top_of: list[str] = field(default_factory=list)
    bottom_of: list[str] = field(default_factory=list)
    generated: bool = False


@dataclass
class SectionGeometry:
    """Section line geometry: ``vertices`` (N, 2) in (u, v); ``chunks`` (M, 2) of (offset, count)."""

    vertices: FloatArray
    chunks: IntArray
    volumes: list[SectionVolume]
    lines: list[SectionLine]

    def chunk_vertices(self, chunk: int, reversed_: bool = False) -> FloatArray:
        offset, count = self.chunks[chunk]
        points = self.vertices[offset : offset + count]
        return points[::-1] if reversed_ else points

    def ring(self, volume: SectionVolume) -> FloatArray:
        """Closed boundary (first vertex repeated at the end) of ``volume``."""
        pieces = [self.chunk_vertices(chunk, reversed_) for chunk, reversed_ in volume.parts]
        return np.concatenate([pieces[0], *(piece[1:] for piece in pieces[1:])])

    def line_vertices(self, line: SectionLine) -> FloatArray:
        pieces = [self.chunk_vertices(chunk) for chunk in line.parts]
        return np.concatenate([pieces[0], *(piece[1:] for piece in pieces[1:])])


@dataclass
class _Run:
    """Part of a unit with positive thickness, between vertex indices of its top and bottom lines."""

    top_start: int
    top_end: int
    bottom_start: int
    bottom_end: int


@dataclass
class _UnitPlan:
    key: str
    top: int
    bottom: int
    runs: list[_Run]


class _LineStore:
    def __init__(self) -> None:
        self.arrays: list[FloatArray] = []
        self.lines: list[SectionLine] = []
        self.breaks: list[set[int]] = []
        self._ids: dict[bytes, int] = {}

    def add(self, line: FloatArray, generated: bool) -> int:
        key = np.ascontiguousarray(line, dtype=np.float64).tobytes()
        index = self._ids.get(key)
        if index is None:
            index = self._ids[key] = len(self.arrays)
            self.arrays.append(line)
            self.lines.append(SectionLine(parts=[], generated=generated))
            self.breaks.append({0, len(line) - 1})
        return index


def build_section_geometry(
    units: Sequence[SectionUnit],
    *,
    top_elevation: Optional[float] = None,
    base_elevation: Optional[float] = None,
) -> SectionGeometry:
    """Build shared section line geometry and one or more polygons per unit.

    Identical lines (e.g. the base of one unit and the top of the unit below) are stored once, and
    split into chunks at every point where any unit using them starts or ends, so neighbouring
    polygons reference the same chunks along their shared contact.

    Top and bottom of a unit must span exactly the same chainage range, and the top must not be
    below the bottom anywhere (tested exactly, without tolerance). Where top and bottom have the
    same vertex chainages, the unit is split into separate polygons wherever it pinches out to
    zero thickness, and a unit with zero thickness everywhere has no polygon. With different
    vertex chainages a unit forms one polygon, which may touch itself where it pinches out.

    :param units: Units with their top and/or bottom lines in section coordinates.
    :param top_elevation: Elevation of the top of units without a top line (AGSi: the area
        extends upwards to the model boundary).
    :param base_elevation: Elevation of the bottom of units without a bottom line (AGSi: the area
        extends downwards to the model boundary).
    :raise AGSIInvalidDataError: If a unit has neither line, its top is below its bottom, or it
        has no bottom (top) line and no ``base_elevation`` (``top_elevation``) is given.
    :raise AGSIUnsupportedGeometryError: If the top and bottom of a unit span different chainages.
    """
    store = _LineStore()
    plans: list[_UnitPlan] = []
    for unit in units:
        top, bottom, top_generated, bottom_generated = _unit_lines(unit, top_elevation, base_elevation)
        if top[0, 0] != bottom[0, 0] or top[-1, 0] != bottom[-1, 0]:
            raise AGSIUnsupportedGeometryError(
                f"Unit '{unit.key}': top spans chainage {top[0, 0]} to {top[-1, 0]} but bottom spans "
                f"{bottom[0, 0]} to {bottom[-1, 0]}; top and bottom must span the same chainage range."
            )
        runs = _positive_runs(unit.key, top, bottom)
        plan = _UnitPlan(unit.key, store.add(top, top_generated), store.add(bottom, bottom_generated), runs)
        store.lines[plan.top].top_of.append(unit.key)
        store.lines[plan.bottom].bottom_of.append(unit.key)
        for run in runs:
            store.breaks[plan.top].update((run.top_start, run.top_end))
            store.breaks[plan.bottom].update((run.bottom_start, run.bottom_end))
        plans.append(plan)

    vertices: list[FloatArray] = []
    chunks: list[tuple[int, int]] = []
    chunk_at: list[dict[int, int]] = []  # per line: break index -> chunk starting there
    offset = 0
    for line_id, array in enumerate(store.arrays):
        vertices.append(array)
        breaks = sorted(store.breaks[line_id])
        starts: dict[int, int] = {}
        for start, end in zip(breaks, breaks[1:]):
            starts[start] = len(chunks)
            chunks.append((offset + start, end - start + 1))
        store.lines[line_id].parts = list(starts.values())
        chunk_at.append(starts)
        offset += len(array)

    edge_ids: dict[bytes, int] = {}

    def edge(upper: FloatArray, lower: FloatArray) -> int:
        nonlocal offset
        segment = np.vstack([upper, lower])
        key = segment.tobytes()
        if key not in edge_ids:
            edge_ids[key] = len(chunks)
            vertices.append(segment)
            chunks.append((offset, 2))
            offset += 2
        return edge_ids[key]

    def parts_between(line_id: int, start: int, end: int) -> list[int]:
        return [chunk for index, chunk in chunk_at[line_id].items() if start <= index < end]

    volumes: list[SectionVolume] = []
    for plan in plans:
        top, bottom = store.arrays[plan.top], store.arrays[plan.bottom]
        for run in plan.runs:
            parts = [(chunk, False) for chunk in parts_between(plan.top, run.top_start, run.top_end)]
            top_right, bottom_right = top[run.top_end], bottom[run.bottom_end]
            if not np.array_equal(top_right, bottom_right):
                parts.append((edge(top_right, bottom_right), False))
            bottom_parts = parts_between(plan.bottom, run.bottom_start, run.bottom_end)
            parts.extend((chunk, True) for chunk in reversed(bottom_parts))
            top_left, bottom_left = top[run.top_start], bottom[run.bottom_start]
            if not np.array_equal(top_left, bottom_left):
                parts.append((edge(top_left, bottom_left), True))
            volumes.append(SectionVolume(key=plan.key, parts=parts))

    return SectionGeometry(
        vertices=np.concatenate(vertices) if vertices else np.empty((0, 2)),
        chunks=np.asarray(chunks, dtype=np.int64).reshape(-1, 2),
        volumes=volumes,
        lines=store.lines,
    )


def _unit_lines(
    unit: SectionUnit, top_elevation: Optional[float], base_elevation: Optional[float]
) -> tuple[FloatArray, FloatArray, bool, bool]:
    top = None if unit.top is None else profile_to_section(unit.top, f"Unit '{unit.key}' top")
    bottom = None if unit.bottom is None else profile_to_section(unit.bottom, f"Unit '{unit.key}' bottom")
    if top is not None and bottom is not None:
        return top, bottom, False, False
    if top is not None:
        if base_elevation is None:
            raise AGSIInvalidDataError(f"Unit '{unit.key}' has no bottom line; a base elevation is required.")
        return top, _horizontal(top, base_elevation), False, True
    if bottom is not None:
        if top_elevation is None:
            raise AGSIInvalidDataError(f"Unit '{unit.key}' has no top line; a top elevation is required.")
        return _horizontal(bottom, top_elevation), bottom, True, False
    raise AGSIInvalidDataError(f"Unit '{unit.key}' has neither a top nor a bottom line.")


def _horizontal(line: FloatArray, elevation: float) -> FloatArray:
    if not np.isfinite(elevation):
        raise AGSIInvalidDataError(f"Elevation must be finite, got {elevation}.")
    return np.array([[line[0, 0], elevation], [line[-1, 0], elevation]], dtype=np.float64) + 0.0


def _positive_runs(key: str, top: FloatArray, bottom: FloatArray) -> list[_Run]:
    if np.array_equal(top[:, 0], bottom[:, 0]):
        if np.any(top[:, 1] < bottom[:, 1]):
            index = int(np.argmax(top[:, 1] < bottom[:, 1]))
            raise AGSIInvalidDataError(f"Unit '{key}': top is below bottom at chainage {top[index, 0]}.")
        zero = top[:, 1] == bottom[:, 1]
        runs: list[_Run] = []
        start: Optional[int] = None
        for k in range(len(top) - 1):
            if zero[k] and zero[k + 1]:
                continue
            if start is None:
                start = k
            if zero[k + 1]:
                runs.append(_Run(start, k + 1, start, k + 1))
                start = None
        if start is not None:
            runs.append(_Run(start, len(top) - 1, start, len(top) - 1))
        return runs
    if not _check_above(key, top, bottom):
        return []
    return [_Run(0, len(top) - 1, 0, len(bottom) - 1)]


def _check_above(key: str, top: FloatArray, bottom: FloatArray) -> bool:
    """Exactly check that ``top`` is not below ``bottom``; return whether it is above anywhere."""
    above = False
    for sign in (*_signs_against(top, bottom), *(-s for s in _signs_against(bottom, top))):
        if sign < 0:
            raise AGSIInvalidDataError(f"Unit '{key}': top line crosses below the bottom line.")
        above = above or sign > 0
    return above


def _signs_against(line: FloatArray, points: FloatArray) -> list[int]:
    """Sign of line(u) - v for each point (u, v), with ``line`` linearly interpolated, exactly."""
    segment = np.clip(np.searchsorted(line[:, 0], points[:, 0], side="right") - 1, 0, len(line) - 2)
    signs = []
    for (u, v), k in zip(points, segment):
        u0, v0 = (Fraction(float(c)) for c in line[k])
        u1, v1 = (Fraction(float(c)) for c in line[k + 1])
        u, v = Fraction(float(u)), Fraction(float(v))
        value = (v0 - v) * (u1 - u0) + (v1 - v0) * (u - u0)
        signs.append((value > 0) - (value < 0))
    return signs


@dataclass
class ModelSection:
    """Section geometry of one AGSi model, with the elements its volume keys refer to."""

    frame: SectionFrame
    geometry: SectionGeometry
    elements: dict[str, AgsiModelElement]


def build_model_section(
    package: AgsiPackage,
    model: AgsiModel,
    *,
    frame: Optional[SectionFrame] = None,
    top_elevation: Optional[float] = None,
    base_elevation: Optional[float] = None,
) -> ModelSection:
    """Build the section geometry of the ``agsiGeometryAreaFromLines`` elements of a 2D section model.

    :param frame: Section placement. Defaults to the model's straight alignment (via
        ``alignmentID``), or :meth:`SectionFrame.local` if the model has none. An explicit frame
        overrides the placement only; a linked alignment must still be straight.
    :param top_elevation: See :func:`build_section_geometry`; defaults to the model boundary
        ``topElevation``.
    :param base_elevation: See :func:`build_section_geometry`; defaults to the model boundary
        ``bottomElevation``.
    :raise AGSIUnsupportedGeometryError: If the linked alignment is not straight, even when
        ``frame`` is given.
    :return: Volume keys are the element IDs (``"element-<index>"`` for elements without one).
        Elements with other geometry types are skipped with a warning.
    """
    # The linked alignment is always validated, so an explicit frame cannot bypass the straight-only rule.
    alignment_frame = section_frame_for_model(package, model)
    if frame is None:
        frame = alignment_frame or SectionFrame.local()
    boundary = model.agsi_model_boundary
    if boundary is not None:
        top_elevation = boundary.top_elevation if top_elevation is None else top_elevation
        base_elevation = boundary.bottom_elevation if base_elevation is None else base_elevation

    profiles: dict[tuple[str, Optional[str]], FloatArray] = {}

    def read_profile(geometry: Optional[AgsiGeometryFromFile]) -> Optional[FloatArray]:
        if geometry is None:
            return None
        cache_key = (geometry.file_uri, geometry.file_part)
        if cache_key not in profiles:
            name = f"Profile '{geometry.file_uri}'"
            data = package.read_geometry(geometry)
            check_lines_only(data, name)
            profiles[cache_key] = profile_to_section(data.single_line(), name)
            _warn_outside_alignment(frame, profiles[cache_key], name)
        return profiles[cache_key]

    units: list[SectionUnit] = []
    elements: dict[str, AgsiModelElement] = {}
    for index, element in enumerate(model.agsi_model_element):
        key = element.element_id or f"element-{index}"
        geometry = element.agsi_geometry
        if not isinstance(geometry, AgsiGeometryAreaFromLines):
            logger.warning(f"Element '{key}': {element.geometry_object_name} is not a section area; skipped.")
            continue
        if key in elements:
            raise AGSIInvalidDataError(f"Duplicate elementID '{key}' in model '{model.model_id}'.")
        elements[key] = element
        units.append(
            SectionUnit(key, read_profile(geometry.agsi_geometry_top), read_profile(geometry.agsi_geometry_bottom))
        )

    geometry_out = build_section_geometry(units, top_elevation=top_elevation, base_elevation=base_elevation)
    return ModelSection(frame=frame, geometry=geometry_out, elements=elements)


def _warn_outside_alignment(frame: SectionFrame, line: FloatArray, name: str) -> None:
    end = frame.end_chainage
    if end is not None and (line[0, 0] < frame.start_chainage or line[-1, 0] > end):
        logger.warning(
            f"{name} spans chainage {line[0, 0]} to {line[-1, 0]}, beyond the alignment "
            f"({frame.start_chainage} to {end}); it is placed on the extension of the straight alignment."
        )
