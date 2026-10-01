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

import zipfile
from pathlib import Path
from typing import Callable, Optional

import pytest

DATA_DIR = Path(__file__).parent / "data"
# Trimmed excerpts of the published Silvertown 2D examples converted to AGSi 1.0.1: an NB section
# model linked by alignmentID to the alignment of a separate plan model, with WKT or DXF profiles.
WKT_PACKAGE_DIR = DATA_DIR / "silvertown_mini_wkt"
DXF_PACKAGE_DIR = DATA_DIR / "silvertown_mini_dxf"
V1_PACKAGE_DIR = DATA_DIR / "agsi_v1_minimal"


def zip_folder(folder: Path, target: Path, rename: Optional[Callable[[str, bytes], tuple[str, bytes]]] = None) -> Path:
    """Zip ``folder`` into ``target``; ``rename`` may change each member's name and content."""
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            name, data = path.relative_to(folder).as_posix(), path.read_bytes()
            if rename is not None:
                name, data = rename(name, data)
            archive.writestr(name, data)
    return target


@pytest.fixture
def wkt_dir() -> Path:
    return WKT_PACKAGE_DIR


@pytest.fixture
def dxf_dir() -> Path:
    return DXF_PACKAGE_DIR


@pytest.fixture
def v1_dir() -> Path:
    return V1_PACKAGE_DIR


@pytest.fixture
def wkt_zip(tmp_path: Path) -> Path:
    return zip_folder(WKT_PACKAGE_DIR, tmp_path / "silvertown_mini_wkt.agsi")


@pytest.fixture
def dxf_zip(tmp_path: Path) -> Path:
    return zip_folder(DXF_PACKAGE_DIR, tmp_path / "silvertown_mini_dxf.agsi")
