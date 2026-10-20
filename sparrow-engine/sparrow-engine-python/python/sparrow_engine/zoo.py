"""Model zoo access: list the Sparrow Model Zoo catalog and download models.

Python port of ``scripts/download_models.sh`` (same selection rules, same
on-disk layout, same ``.sparrow_zip_md5`` install stamp), using only the
standard library. The catalog is read from the pinned Zenodo record, which
publishes ``catalog.toml`` next to the model ZIPs, so the package carries no
copy of the catalog that could drift.

Typical use::

    import sparrow_engine as spe

    spe.download_models(["MDV6-yolov10-e"])     # one model
    spe.download_models()                       # default desktop ONNX set
    for m in spe.zoo_models():                  # browse the catalog
        print(m.id, m.license, m.downloadable)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import tomllib
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

__all__ = [
    "ZooModel",
    "ZooError",
    "zoo_models",
    "download_models",
    "download_model",
    "ZENODO_RECORD",
]

# Zenodo record of the model zoo this package release targets. Must equal
# `[zenodo] record` in sparrow-engine/scripts/catalog.toml (enforced by
# tests/test_zoo.py). Override per call with `record=` or via `ZENODO_RECORD`.
ZENODO_RECORD = "22711452"

_HOSTED = frozenset({"hosted", "hosted_restricted"})
_INSTALL_MARKERS = ("manifest.toml", "pipeline.toml", "ensemble.toml")
_STAMP = ".sparrow_zip_md5"
_USER_AGENT = "sparrow-engine-python (model-zoo downloader)"
_CHUNK = 1 << 20


class ZooError(RuntimeError):
    """Raised for catalog, selection, network, or integrity failures."""


@dataclass(frozen=True)
class ZooModel:
    """One catalog entry of the Sparrow Model Zoo."""

    id: str
    domain: str
    task: str
    format: str
    license: str
    status: str
    hosting_status: str
    zip: str
    family: tuple[str, ...] = ()
    alias: tuple[str, ...] = ()
    flavor: Optional[str] = None
    source_url: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def downloadable(self) -> bool:
        """True when Sparrow distributes the payload (active and hosted)."""
        return self.status == "active" and self.hosting_status in _HOSTED

    @property
    def default(self) -> bool:
        """True when included in the no-argument default download set."""
        return self.downloadable and self.format == "onnx" and not self.flavor

    @classmethod
    def _from_toml(cls, m: dict[str, Any]) -> "ZooModel":
        return cls(
            id=m["id"],
            domain=m.get("domain", ""),
            task=m.get("task", ""),
            format=m.get("format", ""),
            license=m.get("license", ""),
            status=m.get("status", "unpublished"),
            hosting_status=m.get("hosting_status", "pending_rights"),
            zip=m.get("zip", ""),
            family=tuple(m.get("family", ())),
            alias=tuple(m.get("alias", ())),
            flavor=m.get("flavor"),
            source_url=m.get("original_source_url"),
            raw=dict(m),
        )


@dataclass
class _Catalog:
    record: str
    version: str
    models: list[ZooModel]


def _log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


def _urlopen(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    return urllib.request.urlopen(req, timeout=60)  # noqa: S310 — https Zenodo URLs only


def _fetch_bytes(url: str) -> bytes:
    try:
        with _urlopen(url) as resp:
            return resp.read()
    except OSError as exc:
        raise ZooError(f"failed to fetch {url}: {exc}") from exc


def _fetch_to_file(url: str, path: Path) -> str:
    """Stream ``url`` to ``path``; return the hex MD5 of the bytes written."""
    md5 = hashlib.md5()  # noqa: S324 — integrity check against Zenodo's published MD5
    try:
        with _urlopen(url) as resp, open(path, "wb") as out:
            while chunk := resp.read(_CHUNK):
                md5.update(chunk)
                out.write(chunk)
    except OSError as exc:
        raise ZooError(f"failed to download {url}: {exc}") from exc
    return md5.hexdigest()


def _resolve_record(record: Optional[str]) -> str:
    return str(record or os.environ.get("ZENODO_RECORD") or ZENODO_RECORD)


def _record_checksums(record: str) -> dict[str, str]:
    """Map file name -> MD5 from the Zenodo record API (empty if none published)."""
    raw = _fetch_bytes(f"https://zenodo.org/api/records/{record}")
    try:
        files = json.loads(raw).get("files", [])
    except ValueError as exc:
        raise ZooError(f"Zenodo record {record}: invalid API response") from exc
    out = {}
    for f in files:
        ck = f.get("checksum", "")
        if ck.startswith("md5:"):
            out[f["key"]] = ck.split(":", 1)[1]
    return out


def _parse_catalog(data: bytes, record: str, source: str) -> _Catalog:
    """Parse catalog bytes; an empty ``record`` falls back to the catalog's own record."""
    try:
        doc = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ZooError(f"invalid catalog at {source}: {exc}") from exc
    zen = doc.get("zenodo", {})
    return _Catalog(
        record=record or str(zen.get("record") or ZENODO_RECORD),
        version=str(zen.get("version", "?")),
        models=[ZooModel._from_toml(m) for m in doc.get("model", [])],
    )


def _local_catalog_path(catalog: Optional[Union[str, Path]]) -> Optional[Path]:
    local = catalog or os.environ.get("SPARROW_CATALOG")
    return Path(local).expanduser() if local else None


def _load_catalog(
    catalog: Optional[Union[str, Path]],
    record: Optional[str],
    checksums: Optional[dict[str, str]],
) -> _Catalog:
    """Catalog order: ``catalog=`` > ``SPARROW_CATALOG`` > the Zenodo record's catalog.toml.

    A local catalog supplies its own ``[zenodo] record`` unless ``record=`` or
    ``ZENODO_RECORD`` overrides it (same rule as download_models.sh). The
    remote catalog is MD5-checked when ``checksums`` is given.
    """
    path = _local_catalog_path(catalog)
    override = record or os.environ.get("ZENODO_RECORD") or ""
    if path is not None:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ZooError(f"catalog not found at {path}: {exc}") from exc
        return _parse_catalog(data, str(override), str(path))

    rec = _resolve_record(record)
    url = f"https://zenodo.org/records/{rec}/files/catalog.toml"
    data = _fetch_bytes(url)
    if checksums is not None:
        expected = checksums.get("catalog.toml")
        if expected and hashlib.md5(data).hexdigest() != expected:  # noqa: S324
            raise ZooError(f"catalog.toml from record {rec} failed its MD5 check")
    return _parse_catalog(data, rec, url)


def zoo_models(
    *,
    catalog: Optional[Union[str, Path]] = None,
    record: Optional[str] = None,
) -> list[ZooModel]:
    """Return every catalog entry, sorted by (domain, task, id).

    Use ``m.downloadable`` to see which entries Sparrow distributes and
    ``m.default`` for the set fetched by ``download_models()`` with no ids.
    Link-only and pending-rights entries carry ``m.source_url`` instead.
    Installed models are listed by :func:`sparrow_engine.list_models`.
    """
    cat = _load_catalog(catalog, record, None)
    return sorted(cat.models, key=lambda m: (m.domain, m.task, m.id))


def _select(
    models: list[ZooModel], ids: Optional[Iterable[str]], all_models: bool, quiet: bool
) -> list[ZooModel]:
    if ids is None:
        if all_models:
            return [m for m in models if m.downloadable]
        return [m for m in models if m.default]

    by_id = {m.id: m for m in models}
    alias = {a: m.id for m in models for a in m.alias}
    chosen: list[ZooModel] = []
    unknown: list[str] = []
    errors: list[str] = []
    for s in ids:
        if s in by_id:
            rid = s
        elif s in alias:
            rid = alias[s]
            _log(f"note: '{s}' is an old id; resolving to '{rid}'", quiet)
        else:
            unknown.append(s)
            continue
        m = by_id[rid]
        if m.status != "active":
            errors.append(
                f"{m.id} is a {m.status} catalog entry and is not present in the "
                "published model-zoo record."
            )
        elif m.hosting_status not in _HOSTED:
            errors.append(
                f"{m.id} is {m.hosting_status}; Sparrow does not distribute its model "
                f"payload. Original source or contact route: {m.source_url or '<unavailable>'}"
            )
        elif m not in chosen:
            chosen.append(m)
    if unknown:
        raise ZooError(
            "unknown model id(s): " + ", ".join(unknown) + " (see sparrow_engine.zoo_models())"
        )
    if errors:
        raise ZooError("\n".join(errors))
    return chosen


def _installed(model_dir: Path) -> bool:
    return any((model_dir / name).is_file() for name in _INSTALL_MARKERS)


def _install_one(
    m: ZooModel,
    dest: Path,
    record: str,
    expected_md5: Optional[str],
    verify: bool,
    force: bool,
    quiet: bool,
) -> Path:
    target = dest / m.id
    _log(f"==> {m.id}", quiet)

    if not force and _installed(target):
        if verify and expected_md5:
            try:
                stamped = (target / _STAMP).read_text().strip()
            except OSError:
                stamped = ""
            if stamped == expected_md5:
                _log("  already present (checksum up to date); skipping.", quiet)
                return target
            if not stamped:
                _log(
                    "  already present but not checksum-stamped; skipping "
                    "(force=True re-downloads).",
                    quiet,
                )
                return target
            _log(
                f"  local copy is stale (stamped {stamped}, catalog {expected_md5}); "
                "re-downloading.",
                quiet,
            )
        else:
            _log("  already present; skipping (checksum not verified).", quiet)
            return target

    if verify and not expected_md5:
        raise ZooError(
            f"no MD5 published for {m.zip}; refusing to download unverified bytes "
            "(pass verify=False to skip integrity checks)"
        )

    url = f"https://zenodo.org/records/{record}/files/{m.zip}"
    stage = Path(tempfile.mkdtemp(prefix=".stage.", dir=dest))
    try:
        zip_path = stage / m.zip
        _log(f"  downloading {m.zip} ...", quiet)
        actual = _fetch_to_file(url, zip_path)
        if verify:
            if actual != expected_md5:
                raise ZooError(
                    f"MD5 mismatch for {m.zip} (expected {expected_md5}, got {actual}); "
                    "download is corrupt or tampered"
                )
            _log("  [OK] MD5 verified", quiet)
        unpack = stage / "unpacked"
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(unpack)
        except zipfile.BadZipFile as exc:
            raise ZooError(f"{m.zip} is not a valid ZIP archive: {exc}") from exc
        unpacked = unpack / m.id
        if not unpacked.is_dir():
            raise ZooError(f"{m.zip} did not contain the expected '{m.id}/' directory")
        if target.exists():
            shutil.rmtree(target)
        os.replace(unpacked, target)
        if verify:
            (target / _STAMP).write_text(expected_md5 + "\n")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    _log(f"  installed -> {target}", quiet)
    return target


def download_models(
    models: Optional[Union[str, Iterable[str]]] = None,
    *,
    dest: Optional[Union[str, Path]] = None,
    all_models: bool = False,
    force: bool = False,
    verify: bool = True,
    catalog: Optional[Union[str, Path]] = None,
    record: Optional[str] = None,
    quiet: bool = False,
) -> list[Path]:
    """Download models from the Sparrow Model Zoo (Zenodo) and install them.

    Args:
        models: Model id(s) or old alias(es). ``None`` selects the default set
            (hosted desktop ONNX models without a precision flavor).
        dest: Model directory. Defaults to ``SPARROW_ENGINE_MODEL_DIR`` or
            ``~/.sparrow-engine/models`` — the directory the engine reads by
            default. If you call ``init(model_dir=...)``, pass the same path.
        all_models: With ``models=None``, select every hosted runtime package
            (TFLite, cascades, ensembles and precision variants included).
        force: Re-download models that are already installed.
        verify: Check each ZIP against the MD5 published by the Zenodo record
            API. With ``verify=False`` no integrity check is made.
        catalog: Local ``catalog.toml`` instead of the record's published one
            (also ``SPARROW_CATALOG``).
        record: Zenodo record id override (also ``ZENODO_RECORD``).
        quiet: Suppress progress messages on stderr.

    Returns:
        The installed model directories, in selection order.

    Raises:
        ZooError: Unknown id, a link-only / pending-rights / candidate entry,
            a network failure, or a failed integrity check. A failed model
            leaves no partial directory behind.
    """
    if isinstance(models, str):
        models = [models]
    elif models is not None:
        models = list(models)

    checksums: dict[str, str] = {}
    if _local_catalog_path(catalog) is not None:
        cat = _load_catalog(catalog, record, None)
        if verify:
            checksums = _record_checksums(cat.record)
    else:
        if verify:
            checksums = _record_checksums(_resolve_record(record))
        cat = _load_catalog(None, record, checksums if verify else None)
    rec = cat.record

    selected = _select(cat.models, models, all_models, quiet)
    if dest is None:
        from sparrow_engine import _default_model_dir

        dest = _default_model_dir()
    out_dir = Path(dest).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(
        f"Sparrow Model Zoo v{cat.version} — Zenodo record {rec}; "
        f"{len(selected)} model(s) -> {out_dir}",
        quiet,
    )
    return [
        _install_one(m, out_dir, rec, checksums.get(m.zip), verify, force, quiet) for m in selected
    ]


def download_model(model: str, **kwargs: Any) -> Path:
    """Download one model; see :func:`download_models` for keyword arguments."""
    return download_models([model], **kwargs)[0]
