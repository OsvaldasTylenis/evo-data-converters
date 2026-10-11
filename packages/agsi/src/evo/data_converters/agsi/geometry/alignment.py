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

"""Straight section alignments and the section (u, v) <-> world (x, y, z) transform."""

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from evo_schemas.components import Rotation_V1_1_0

from ..reader.errors import AGSIInvalidDataError, AGSIUnsupportedGeometryError
from ..reader.geometry_files import GeometryData, chain_lines
from ..reader.models import AgsiDocument, AgsiModel, AgsiModelAlignment
from ..reader.package import AgsiPackage

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SectionFrame:
    """Vertical section plane along a straight alignment.

    Section coordinates are ``u`` = chainage and ``v`` = elevation, so AGSi 2D profile
    coordinates (chainage, elevation) are used unchanged. The section origin, (u, v) = (0, 0), is
    the point at chainage 0 and elevation 0, which is before the first alignment vertex when
    ``start_chainage`` is positive.

    :param origin_xy: World XY of the first alignment vertex.
    :param direction_xy: Unit horizontal vector from the first towards the last alignment vertex.
    :param start_chainage: Chainage of the first alignment vertex (AGSi ``startChainage``).
    :param length: Horizontal length of the alignment, if known.
    """

    origin_xy: tuple[float, float]
    direction_xy: tuple[float, float]
    start_chainage: float = 0.0
    length: Optional[float] = None

    @classmethod
    def local(cls) -> "SectionFrame":
        """Frame for a section without a known alignment: chainage 0 at (0, 0), increasing to +X (east)."""
        return cls(origin_xy=(0.0, 0.0), direction_xy=(1.0, 0.0))

    @property
    def origin(self) -> tuple[float, float, float]:
        """World XYZ of section coordinates (u, v) = (0, 0), i.e. chainage 0 at elevation 0."""
        x, y = self.chainage_to_xy(0.0)[0]
        return (float(x), float(y), 0.0)

    @property
    def bearing(self) -> float:
        """Direction of increasing chainage, in degrees clockwise from north (+Y), in [0, 360)."""
        return math.degrees(math.atan2(self.direction_xy[0], self.direction_xy[1])) % 360.0

    @property
    def rotation(self) -> Rotation_V1_1_0:
        """Evo rotation of the section plane: local u along the alignment, v up, w = u x v.

        Same convention as ``evo.data_converters.common.utils.convert_rotation``.
        """
        return Rotation_V1_1_0(dip_azimuth=(self.bearing + 90.0) % 360.0, dip=90.0, pitch=180.0)

    @property
    def end_chainage(self) -> Optional[float]:
        return None if self.length is None else self.start_chainage + self.length

    def chainage_to_xy(self, chainage: npt.ArrayLike) -> FloatArray:
        """World XY, shape (N, 2), of positions at the given chainages along the alignment."""
        along = np.asarray(chainage, dtype=np.float64).reshape(-1) - self.start_chainage
        xy: FloatArray = np.asarray(self.origin_xy) + np.outer(along, self.direction_xy)
        return xy

    def section_to_world(self, uv: npt.ArrayLike) -> FloatArray:
        """World XYZ, shape (N, 3), of section coordinates (u, v) with shape (N, 2)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        return np.column_stack([self.chainage_to_xy(uv[:, 0]), uv[:, 1]])

    def world_to_section(self, xyz: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Project world XYZ, shape (N, 3), onto the section.

        :return: Section coordinates (u, v), shape (N, 2), and the signed horizontal offset of
            each point from the section plane (positive to the right, looking along the alignment).
        """
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        dx, dy = self.direction_xy
        rel = xyz[:, :2] - np.asarray(self.origin_xy)
        u = rel[:, 0] * dx + rel[:, 1] * dy + self.start_chainage
        offset = rel[:, 0] * dy - rel[:, 1] * dx
        return np.column_stack([u, xyz[:, 2]]), offset


def straight_alignment(
    geometry: Union[GeometryData, Sequence[npt.ArrayLike]], start_chainage: float = 0.0
) -> SectionFrame:
    """Build a :class:`SectionFrame` from straight alignment geometry.

    The alignment must be one connected line (e.g. a WKT LINESTRING, a DXF polyline, or a chain
    of DXF LINE entities) whose XY vertices all lie exactly on the straight line from its first
    to its last vertex, in order. Z values are ignored. Collinearity is tested with exact rational
    arithmetic on the coordinates as written in the file (the shortest decimal representation of
    each parsed value), so no tolerance is applied. Repeated consecutive vertices are ignored.

    :raise AGSIUnsupportedGeometryError: If the alignment is curved, changes direction, folds back
        or is closed, contains arcs or unread DXF entities, or is disconnected or branched.
    :raise AGSIInvalidDataError: If the geometry is not a line, has non-finite coordinates or has
        zero length.
    """
    if isinstance(geometry, GeometryData):
        check_lines_only(geometry, "Alignment")
        lines = geometry.lines
    else:
        lines = [np.asarray(line, dtype=np.float64) for line in geometry]
    lines = [line.reshape(len(line), -1) for line in lines if len(line)]
    if not lines:
        raise AGSIInvalidDataError("Alignment geometry does not contain a line.")

    chains = chain_lines(lines, tolerance=0.0)
    if len(chains) != 1:
        raise AGSIUnsupportedGeometryError(
            f"Alignment must be a single connected line, but its parts form {len(chains)} separate or branched lines."
        )
    if chains[0].shape[1] not in (2, 3) or not np.all(np.isfinite(chains[0][:, :2])):
        raise AGSIInvalidDataError("Alignment geometry must have finite X and Y coordinates.")
    xy = chains[0][:, :2]
    keep = np.concatenate([[True], np.any(xy[1:] != xy[:-1], axis=1)])
    xy = xy[keep]
    if len(xy) < 2:
        raise AGSIInvalidDataError("Alignment has zero length.")

    _check_straight(xy)
    start, end = xy[0], xy[-1]
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    return SectionFrame(
        origin_xy=(float(start[0]), float(start[1])),
        direction_xy=(float((end[0] - start[0]) / length), float((end[1] - start[1]) / length)),
        start_chainage=float(start_chainage),
        length=length,
    )


def _exact(value: float) -> Fraction:
    # repr() gives the shortest decimal that round-trips, i.e. the value as written in WKT/DXF text.
    return Fraction(repr(float(value)))


def _check_straight(xy: FloatArray) -> None:
    points = [(_exact(x), _exact(y)) for x, y in xy]
    (x0, y0), (x1, y1) = points[0], points[-1]
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        raise AGSIUnsupportedGeometryError("Alignment is closed: its first and last vertices are the same point.")
    previous = Fraction(0)
    for i, (x, y) in enumerate(points[1:], start=1):
        if dx * (y - y0) - dy * (x - x0) != 0:
            raise AGSIUnsupportedGeometryError(
                f"Alignment is not straight: vertex {i} ({float(x)}, {float(y)}) is not on the line between its "
                "first and last vertices. Only straight alignments are supported; curved alignments and "
                "alignments with direction changes are not."
            )
        along = dx * (x - x0) + dy * (y - y0)
        if along <= previous:
            raise AGSIUnsupportedGeometryError(
                f"Alignment is not straight: it reverses direction at vertex {i} ({float(x)}, {float(y)})."
            )
        previous = along


def check_lines_only(geometry: GeometryData, what: str) -> None:
    """Reject geometry with arcs, unread DXF entities, or anything other than lines.

    :raise AGSIUnsupportedGeometryError: For DXF arcs (polyline bulges) or unread DXF entities,
        which would otherwise be lost or replaced by straight chords.
    :raise AGSIInvalidDataError: If the geometry contains polygons, points or a mesh.
    """
    if geometry.curved_segments:
        raise AGSIUnsupportedGeometryError(
            f"{what} contains {geometry.curved_segments} curved (arc) polyline segment(s); curves are not supported."
        )
    if geometry.skipped_entities:
        raise AGSIUnsupportedGeometryError(
            f"{what} contains unsupported DXF entities {geometry.skipped_entities}; only lines and polylines are read."
        )
    if geometry.polygons or geometry.points or geometry.mesh is not None:
        raise AGSIInvalidDataError(f"{what} geometry must contain only lines.")


def find_model_alignment(document: AgsiDocument, model: AgsiModel) -> Optional[AgsiModelAlignment]:
    """Return the ``agsiModelAlignment`` referenced by ``model.alignmentID``, or ``None`` if unset.

    The alignment may be defined in any model of the document (typically a separate plan model).

    :raise AGSIInvalidDataError: If no alignment, or more than one, has that ``alignmentID``.
    """
    if not model.alignment_id:
        return None
    matches = [
        alignment
        for candidate in document.agsi_model
        for alignment in candidate.agsi_model_alignment
        if alignment.alignment_id == model.alignment_id
    ]
    if not matches:
        raise AGSIInvalidDataError(f"alignmentID '{model.alignment_id}' does not match any agsiModelAlignment.")
    if len(matches) > 1:
        raise AGSIInvalidDataError(f"alignmentID '{model.alignment_id}' matches {len(matches)} agsiModelAlignment.")
    return matches[0]


def section_frame_for_model(package: AgsiPackage, model: AgsiModel) -> Optional[SectionFrame]:
    """Resolve ``model.alignmentID`` to its alignment and return its straight :class:`SectionFrame`.

    :return: ``None`` if the model has no ``alignmentID``.
    :raise AGSIUnsupportedGeometryError: If the alignment is not straight (see :func:`straight_alignment`).
    """
    alignment = find_model_alignment(package.document, model)
    if alignment is None:
        return None
    if alignment.agsi_geometry is None:
        raise AGSIInvalidDataError(f"agsiModelAlignment '{alignment.alignment_id}' has no agsiGeometry.")
    try:
        return straight_alignment(
            package.read_geometry(alignment.agsi_geometry), start_chainage=alignment.start_chainage or 0.0
        )
    except AGSIInvalidDataError as e:
        raise type(e)(f"agsiModelAlignment '{alignment.alignment_id}': {e}") from e
