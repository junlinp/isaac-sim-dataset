"""Mirror only the dependency closure of NVIDIA's published Office USD."""

import concurrent.futures
import json
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit, unquote
import requests
from pxr import Sdf, UsdUtils

from settings import ASSET_CACHE

ROOT = ASSET_CACHE / "cloud"
HOST = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
ENTRY = HOST + "/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd"
BUILTINS = {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl", "OmniPBR_ClearCoat.mdl"}


def local(url):
    p = (ROOT / unquote(urlsplit(url).path.lstrip("/"))).resolve()
    assert p.is_relative_to(ROOT.resolve()), url
    return p


def download(url):
    p = local(url)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.stat().st_size:
        return url, p, None
    for attempt in range(3):
        try:
            with requests.get(url, timeout=(20, 120), stream=True) as r:
                r.raise_for_status()
                with p.with_suffix(p.suffix + ".part").open("wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
            p.with_suffix(p.suffix + ".part").replace(p)
            return url, p, None
        except requests.RequestException as e:
            if attempt == 2:
                return url, p, str(e)
            time.sleep(attempt + 1)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    seen, pending, failed, skipped = set(), {ENTRY}, {}, set()
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        while pending:
            batch = sorted(pending - seen)
            pending = set()
            for url, path, error in pool.map(download, batch):
                seen.add(url)
                if error:
                    failed[url] = error
                    print("ASSET_ERROR", url, error, flush=True)
                    continue
                total += path.stat().st_size
                refs = set()
                if path.suffix.lower() in [".usd", ".usda", ".usdc"]:
                    layer = Sdf.Layer.FindOrOpen(str(path))
                    assert layer is not None, path
                    refs.update(layer.GetExternalReferences())
                    refs.update(layer.GetExternalAssetDependencies())
                    for group in UsdUtils.ExtractExternalReferences(str(path)):
                        refs.update(group)
                elif path.suffix.lower() == ".mdl":
                    refs.update(
                        re.findall(
                            r'"([^"\n]+\.(?:png|jpg|jpeg|exr|hdr|dds|tif|mdl))"',
                            path.read_text(errors="replace"),
                            re.I,
                        )
                    )
                    # MDL relative modules use namespace syntax rather than asset strings.
                    for module in re.findall(
                        r"(?:using|import)\s+((?:\.{1,2}::)[A-Za-z0-9_:]+)",
                        path.read_text(errors="replace"),
                    ):
                        parts = module.split("::")
                        prefix = parts.pop(0)
                        relative = ("../" if prefix == ".." else "./") + "/".join(parts)
                        if relative.endswith("/*"):
                            relative = relative[:-2]
                        refs.add(relative + ".mdl")
                for ref in refs:
                    if ref in BUILTINS or ref.startswith("::"):
                        skipped.add(ref)
                        continue
                    dependency = urljoin(url, ref)
                    assert (
                        urlsplit(dependency).netloc == urlsplit(HOST).netloc
                    ), dependency
                    if dependency not in seen:
                        pending.add(dependency)
            print(
                "CACHE_PROGRESS",
                len(seen),
                "next",
                len(pending),
                "bytes",
                total,
                "failed",
                len(failed),
                flush=True,
            )
    report = dict(
        status="passed" if not failed else "failed",
        provider="NVIDIA Isaac Sim 5.1 Office",
        entry_url=ENTRY,
        local_entry=str(local(ENTRY)),
        files=len(seen),
        bytes=total,
        failed=failed,
        builtin_mdl_modules=sorted(skipped),
    )
    (ROOT.parent / "cache_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    assert not failed


if __name__ == "__main__":
    main()
