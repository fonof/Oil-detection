"""Download Kuwait oil spill SAFE from CDSE or ASF."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "kuwait_2017_oil"
CDSE_ID = "2cd4013a-1337-5858-95a2-3afcc9c477d2"
ASF_URL = (
    "https://datapool.asf.alaska.edu/GRD_HD/SA/"
    "S1A_IW_GRDH_1SDV_20170810T024714_20170810T024738_017855_01DEF7_F48C.zip"
)
APPROVE_ASF = (
    "https://urs.earthdata.nasa.gov/approve_app?client_id=BO_n7nTIlMljdvU6kRRB3g"
)


def download_stream(resp: requests.Response, out: Path) -> None:
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length") or 0)
    print(f"saving -> {out} ({total / 1e6:.0f} MB)")
    n = 0
    with open(out, "wb") as f:
        for chunk in resp.iter_content(1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            n += len(chunk)
            if total and n % (80 * 1024 * 1024) < 1024 * 1024:
                print(f"  {100 * n / total:.0f}% ({n / 1e6:.0f} MB)")
    print(f"done {out.stat().st_size / 1e6:.0f} MB")


def try_cdse(user: str, password: str) -> bool:
    print("Trying CDSE...")
    r = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "client_id": "cdse-public",
            "username": user,
            "password": password,
            "grant_type": "password",
        },
        timeout=60,
    )
    if r.status_code != 200:
        print(f"CDSE auth failed: {r.status_code} {r.text[:200]}")
        return False
    token = r.json()["access_token"]
    url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({CDSE_ID})/$value"
    out = OUT / "S1A_IW_GRDH_1SDV_20170810T024714_20170810T024738_017855_01DEF7_F48C.zip"
    with requests.get(
        url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=180
    ) as resp:
        if resp.status_code != 200:
            print(f"CDSE download failed: {resp.status_code}")
            return False
        download_stream(resp, out)
    return True


def try_asf(user: str, password: str) -> bool:
    print("Trying ASF via asf_search...")
    try:
        import asf_search as asf
    except ImportError:
        print("asf_search not installed")
        return False
    try:
        session = asf.ASFSession().auth_with_creds(user, password)
    except Exception as e:
        print(f"ASF auth failed: {e}")
        print(f"\nОткрой в браузере (залогинься как {user}) и нажми Approve:\n{APPROVE_ASF}\n")
        return False
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    results = asf.granule_search(
        ["S1A_IW_GRDH_1SDV_20170810T024714_20170810T024738_017855_01DEF7_F48C"]
    )
    grd = [r for r in results if r.properties.get("processingLevel") == "GRD_HD"]
    if not grd:
        print("GRD_HD not found")
        return False
    print("Downloading GRD_HD...")
    asf.download_urls(urls=[grd[0].properties["url"]], path=str(out), session=session)
    return True


def main() -> int:
    user = os.environ.get("EARTHDATA_USERNAME") or os.environ.get("CDSE_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD") or os.environ.get("CDSE_PASSWORD")
    if not user or not password:
        print("Set EARTHDATA_USERNAME / EARTHDATA_PASSWORD")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    # Prefer ASF (smaller ~685MB); CDSE ~1.6GB
    if try_asf(user, password):
        return 0
    if try_cdse(user, password):
        return 0
    print("Both ASF and CDSE failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
