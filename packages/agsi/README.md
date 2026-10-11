<p align="center"><a href="https://seequent.com" target="_blank"><picture><source media="(prefers-color-scheme: dark)" srcset="https://developer.seequent.com/img/seequent-logo-dark.svg" alt="Seequent logo" width="400" /><img src="https://developer.seequent.com/img/seequent-logo.svg" alt="Seequent logo" width="400" /></picture></a></p>
<p align="center">
    <a href="https://pypi.org/project/evo-data-converters-agsi/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-data-converters-agsi" /></a>
    <a href="https://github.com/SeequentEvo/evo-data-converters/actions/workflows/on-merge.yaml"><img src="https://github.com/SeequentEvo/evo-data-converters/actions/workflows/on-merge.yaml/badge.svg" alt="" /></a>
</p>
<p align="center">
    <a href="https://developer.seequent.com/" target="_blank">Seequent Developer Portal</a>
    &bull; <a href="https://community.seequent.com/group/19-evo" target="_blank">Seequent Community</a>
    &bull; <a href="https://seequent.com" target="_blank">Seequent website</a>
</p>

## Evo

Evo is a unified platform for geoscience teams. It enables access, connection, computation, and management of subsurface data. This empowers better decision-making, simplified collaboration, and accelerated innovation. Evo is built on open APIs, allowing developers to build custom integrations and applications. Our open schemas, code examples, and SDK are available for the community to use and extend. 

Evo is powered by Seequent, a Bentley organisation.

## Pre-requisites

* Python virtual environment with Python 3.10, 3.11, or 3.12
* Git

## Installation

`pip install evo-data-converters-agsi`

## agsi converter

Converts [AGSi](https://ags-data-format-wg.gitlab.io/agsi/agsi_standard/latest/) ground models
to and from Evo `geological-sections` objects.

> **Status:** work in progress. This release contains the AGSi **reader**, the 2D **section
> geometry engine** and the **importer** (`convert_agsi`) for straight 2D sections;
> `export_agsi` (export) still raises `NotImplementedError`.

### Importing AGSi into Evo

`convert_agsi` converts each 2D section model of an AGSi 1.0.x package (a model with
`agsiGeometryAreaFromLines` elements) into one Evo `geological-sections` object
(`GeologicalSections_V1_2_0`) and publishes it, or returns it with `publish_objects=False`.

```python
from evo.data_converters.agsi import convert_agsi

objects = convert_agsi(
    "Silvertown2D_wkt_straight_synthetic.agsi",
    evo_workspace_metadata=workspace_metadata,  # or service_manager_widget=...
    coordinate_reference_system="EPSG:27700",
    base_elevation=-100.0,  # base of the lowest unit, which has a top profile only
    upload_path="agsi",
)
```

* **CRS.** The `globalXYSystem` declared in the AGSi file is never applied automatically, because
  it may be inferred rather than stated by the producer. Pass `coordinate_reference_system`; a
  warning is logged when it is missing or differs from the declared EPSG code. Coordinate system
  transformations (`transformShiftX`, ...) are not applied.
* **Elevations.** `base_elevation` / `top_elevation` close units without a bottom / top profile;
  they default to the model boundary's `bottomElevation` / `topElevation`.
* **Errors.** Curved alignments, 3D profiles and files without a 2D section model raise
  `AGSIUnsupportedGeometryError`; a missing required elevation or invalid data raise
  `AGSIInvalidDataError`. Nothing is published when an error is raised.

Object layout:

| Evo | From AGSi |
| --- | --- |
| `name`, the single `sections[0]` and `folders[0]` | `modelName` (else `modelID`); the folder is named after the alignment |
| `sections[0].origin`, `rotation` | the straight alignment at chainage 0 and elevation 0; vertical plane along the alignment |
| `line_geometry` | (chainage, elevation) vertices and chunks; each profile is stored once |
| `volumes` | one or more closed polygons per element, `feature` `OutputVolume`; `material_key` and `layer` are the `elementID`, `name` the `elementName` |
| `surfaces` | one per distinct top/bottom line (`ContactSurface`); generated horizontal lines are `BoundarySurface` |
| `materials`, `layer_order` | one per element (`colourRGB` or a default colour), top to bottom |
| `volume_attributes` | `elementID`, `elementType`, and one attribute per `agsiDataParameterValue` code (and case); numeric values are continuous, others strings |
| `extensions.agsi` | `schemaVersion`, `modelID`, `modelName`, `coordSystemID`, `alignmentID`, `alignmentName`, `startChainage`, `alignmentLength` |

Models without section areas (plan models holding alignments, 3D models of surfaces) are skipped.
`valueProfile` parameters are not converted.

### Reading AGSi packages

`AgsiPackage.open` accepts:

* a `.agsi` (or `.zip`) archive with `index.json`, the main JSON file and its supporting files;
* a folder holding the extracted contents of such an archive;
* the main JSON file itself or `index.json`. Supporting files are resolved relative to its folder.

```python
from evo.data_converters.agsi.reader import AgsiPackage

package = AgsiPackage.open("Silvertown2D_wkt_1.agsi")
print(package.schema_version)  # "1.0.1"

section = package.models[0]
print(package.document.coordinate_system(section))  # agsProjectCoordinateSystem via coordSystemID

for model in package.models:
    for element in model.agsi_model_element:
        print(element.element_id, element.geometry_object_name)
        for role, geometry in element.geometry_files():  # role: "top", "bottom", ...
            data = package.read_geometry(geometry)
            if data.lines:
                vertices = data.single_line()  # (n, 2) chainage/elevation or (n, 3) x/y/z
            elif data.mesh is not None or data.polygons:
                mesh = data.to_triangle_mesh()  # vertices (n, 3), triangles (m, 3)
```

`read_agsi_file(filepath)` in `evo.data_converters.agsi.importer` returns the same `AgsiPackage`.
I/O problems raise `AGSIDataFileIOError`. Malformed documents raise `AGSIInvalidDataError`.

#### Supported AGSi versions

Only **AGSi 1.0.x** is supported, read as-is with no normalisation. The version is taken from
`agsSchema.version`, and any version declared in `index.json` must also be 1.0.x. Versions must be
written in full as `1.0.<patch>` (e.g. `1.0.0`, `1.0.1`); `1.0`, `1.0.`, `1.0.1.2`, prerelease
suffixes and surrounding whitespace are rejected. Other versions,
including the pre-1.0 AGSi v0.6 format (no `agsSchema`), raise `AGSIUnsupportedVersionError`
(a subclass of `AGSIInvalidDataError`).

Pydantic models (`reader.models`) cover the subset used by the converter: `agsSchema`, `agsFile`,
`agsProject` (incl. coordinate systems), `agsiModel`, `agsiModelElement`, `agsiModelBoundary`,
`agsiModelAlignment`, `agsiGeometryFromFile`, `agsiGeometryAreaFromLines`,
`agsiGeometryVolFromSurfaces`, `agsiGeometryPlane`, `agsiGeometryLayer` and
`agsiDataParameterValue`. Unknown attributes are kept (`model.extra_attributes`,
`model.to_agsi()`).

Following the AGSi
[model sections guidance](https://ags-data-format-wg.gitlab.io/agsi/AGSi_Documentation/Guidance_Model_Sections/),
a 2D section model gives its section line with `alignmentID`, which refers to an
`agsiModelAlignment` (usually in a separate plan model). Profile chainages are measured along
that alignment from its `startChainage`.

#### Geometry files

`reader.geometry_files.parse_geometry` returns numpy arrays (`GeometryData`) from:

* **WKT/EWKT**: `LINESTRING`, `POLYGON`, `MULTI*`, `GEOMETRYCOLLECTION` and TINs, 2D or 3D.
  Collections of triangles are returned as an indexed mesh.
* **DXF**: `LINE` chains, `LWPOLYLINE`, `POLYLINE` (incl. polyface meshes), `3DFACE` and `POINT`.
  Connected `LINE` entities (as in the Silvertown profiles) are joined into one polyline.
  `filePart` selects a layer.
* **STL**: ASCII and binary.

The reader tolerates quirks found in published examples:

* The declared `fileFormat` is checked against the file content; the content wins (logged).
* A `fileURI` that is not found exactly is matched case-insensitively (with a warning).
* Numbers split over two lines in Mathematica's "exponent above" style are repaired.

URIs that point outside the package are rejected, as is an `index.json` `filePath` that does
(absolute, drive, UNC or URL paths, and `..` segments that leave the folder holding `index.json`,
including percent-encoded ones). For extracted folders, the resolved path of each file read must
also stay inside the package folder, so symlinks and junctions cannot point outside it. DXF arcs (`ARC`, polyline bulges) and other
unsupported entities are not converted; they are counted in `GeometryData.curved_segments` and
`GeometryData.skipped_entities`.

### Section geometry

`evo.data_converters.agsi.geometry` turns a 2D section model (`agsiGeometryAreaFromLines`
elements) into section-plane geometry for Evo `geological-sections`; `convert_agsi` uses it.

```python
from evo.data_converters.agsi.geometry import build_model_section

section = build_model_section(package, package.models[0], base_elevation=-100.0)
frame, geometry = section.frame, section.geometry
frame.origin, frame.rotation           # world point at chainage 0, Evo rotation of the section plane
geometry.vertices, geometry.chunks     # (u, v) = (chainage, elevation) vertices; [offset, count] chunks
for volume in geometry.volumes:        # one or more closed polygons per element
    ring = geometry.ring(volume)
for line in geometry.lines:            # each top/bottom profile once, shared by the units above and below
    line.top_of, line.bottom_of, line.generated
frame.section_to_world(geometry.vertices)  # world x, y, z
```

* **Alignment.** The section frame comes from the `agsiModelAlignment` referenced by the
  model's `alignmentID`; without one, chainage is used as a local x axis (`SectionFrame.local()`),
  and you can pass `frame=` explicitly to override the placement. A linked alignment is still
  checked, so `frame=` cannot be used to accept a curved one. Only **straight** alignments are supported. Consecutive
  vertices or connected DXF `LINE`s are merged only when they are *exactly* collinear (exact
  arithmetic on the stored decimal values, no tolerance). Curved, kinked, closed, branched or
  disconnected alignments and DXF arcs raise `AGSIUnsupportedGeometryError` (or
  `AGSIInvalidDataError` for broken data). Nothing is smoothed or simplified.
* **Profiles** must be 2D (chainage, elevation). Duplicate vertices are dropped and profiles
  drawn right to left are reversed; vertical steps and overhangs (chainage not strictly
  increasing) are rejected. Profiles that run past the end of the alignment are only logged.
* **Units.** A unit's top and bottom must cover the same chainage range. Where they coincide
  (pinch-out) the unit is split into separate polygons; a unit with zero thickness everywhere has
  no polygon. A top below its bottom raises `AGSIInvalidDataError`, checked exactly. Pinch-out
  splitting needs top and bottom to share their chainage sampling; otherwise one polygon is made,
  and it may touch itself.
* **Shared contacts.** Identical profiles (such as the base of one unit and the top of the next)
  are stored once and split at the same chainages, so neighbouring polygons share their
  boundaries exactly.
* **Missing top or bottom.** AGSi lets a unit have a top only (it extends down to the next
  surface). Here, a missing bottom becomes a horizontal line at `base_elevation`, or the model
  boundary's `bottomElevation`. A missing top uses `top_elevation` or `topElevation`. If neither
  is given, `AGSIInvalidDataError` is raised. Overlaps with lower units are not trimmed.
* DXF `LINE` end points are joined by the reader within 1e-9 before the exact checks.

### Tests

```shell
cd packages/common && uv run test-agsi
```

The committed fixtures under `tests/data/` are small AGSi 1.0.1 excerpts of the Silvertown
examples. The examples published on the AGSi website are AGSi v0.6, which is not supported. To
also run the integration tests (not committed; up to 80 MB), set `AGSI_SAMPLE_DIR` to a folder
containing AGSi 1.0.1 conversions of them, named like the originals (`Silvertown2D_{wkt,dxf}_{1,3}.agsi`
and `Silvertown3D_{wkt,dxf,stl}.agsi`). Their NB/SB alignments are curved, so the geometry and
importer tests check that they are rejected (the 3D models are rejected for their 3D profiles); an
optional `Silvertown2D_wkt_straight_synthetic.agsi` in the same folder (the same units on a
straight alignment) is built and converted end to end.

All tests are offline: the importer tests build the objects with a local data client that writes
parquet files to a temporary folder, mock the Evo clients and publishing, and fail on any
non-loopback network connection.


## Code of conduct

We rely on an open, friendly, inclusive environment. To help us ensure this remains possible, please familiarise yourself with our [code of conduct.](https://github.com/SeequentEvo/evo-data-converters/blob/main/CODE_OF_CONDUCT.md)

## License
Evo data converters are open source and licensed under the [Apache 2.0 license.](./LICENSE.md)

Copyright © 2026 Bentley Systems, Incorporated.

Licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and