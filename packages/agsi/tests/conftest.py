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

import json
import shutil
import socket
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
from evo.objects.utils.data import ObjectDataClient
from evo.objects.utils.table_formats import get_known_format

DATA_DIR = Path(__file__).parent / "data"
# Trimmed excerpts of the published Silvertown 2D examples converted to AGSi 1.0.1: an NB section
# model linked by alignmentID to the alignment of a separate plan model, with WKT or DXF profiles.
WKT_PACKAGE_DIR = DATA_DIR / "silvertown_mini_wkt"
DXF_PACKAGE_DIR = DATA_DIR / "silvertown_mini_dxf"
V1_PACKAGE_DIR = DATA_DIR / "agsi_v1_minimal"
# Straight replacement for the curved NB alignment of the mini packages (end points of the
# straight synthetic Silvertown example).
STRAIGHT_NB = "LINESTRING (539591.028 179854.986, 539910.921 180226.890)"


def straighten(source: Path, target: Path) -> Path:
    """Copy a mini Silvertown package, replacing its curved NB alignment with a straight one."""
    folder = Path(shutil.copytree(source, target))
    (folder / "Alignments" / "NB_Alignment.txt").write_text(STRAIGHT_NB)
    return folder


def edit_document(folder: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    """Edit the main JSON document of a mini Silvertown package in place."""
    path = folder / "Silvertown_mini.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    edit(document)
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.fixture
def straight_wkt_dir(tmp_path: Path) -> Path:
    return straighten(WKT_PACKAGE_DIR, tmp_path / "straight_wkt")


@pytest.fixture
def local_data_client(tmp_path: Path) -> MagicMock:
    """Offline ``ObjectDataClient``: ``save_table`` writes parquet files to a local cache folder."""
    cache = tmp_path / "cache"
    cache.mkdir()
    client = MagicMock(spec=ObjectDataClient)
    client.cache_location = cache

    def save_table(table: pa.Table) -> dict:
        return dict(get_known_format(table).save_table(table=table, destination=cache))

    client.save_table.side_effect = save_table
    return client


@pytest.fixture
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any attempt to connect to a non-loopback address (asyncio uses loopback sockets internally)."""
    loopback = {"127.0.0.1", "::1", "localhost"}
    connect, connect_ex, create_connection = socket.socket.connect, socket.socket.connect_ex, socket.create_connection

    def check(address: Any) -> None:
        # Non-tuple addresses are local (AF_UNIX) sockets.
        if isinstance(address, tuple) and address[0] not in loopback:
            raise AssertionError(f"Network access to {address!r} is not allowed in these tests.")

    def guarded_connect(self: socket.socket, address: Any) -> None:
        check(address)
        connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        check(address)
        return connect_ex(self, address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        check(address)
        return create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)


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
