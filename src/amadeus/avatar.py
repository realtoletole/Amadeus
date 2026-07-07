"""Live2D avatar support: vendor runtime setup and model discovery.

The Cubism core is Live2D's proprietary-but-freely-usable runtime; we
download it from Live2D's official CDN (never redistribute it) alongside
the open-source pixi renderer stack, into the data dir, so the app stays
fully offline after `python -m amadeus avatar-setup`.

The user supplies their own model (purchased, commissioned, or freely
licensed): a folder containing a ``*.model3.json`` plus its textures,
dropped anywhere under ``<data>/models/avatar/``.
"""

from __future__ import annotations

from pathlib import Path

from .config import Settings

VENDOR_FILES = {
    "live2dcubismcore.min.js":
        "https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js",
    "pixi.min.js":
        "https://cdn.jsdelivr.net/npm/pixi.js@6.5.10/dist/browser/pixi.min.js",
    "cubism4.min.js":
        "https://cdn.jsdelivr.net/npm/pixi-live2d-display@0.4.0/dist/cubism4.min.js",
}


def vendor_dir(settings: Settings) -> Path:
    return settings.data_dir / "vendor"


def avatar_model_dir(settings: Settings) -> Path:
    return settings.data_dir / "models" / "avatar"


def vendor_ready(settings: Settings) -> bool:
    return all((vendor_dir(settings) / name).exists() for name in VENDOR_FILES)


def find_model3(settings: Settings) -> Path | None:
    """First Cubism 4 model definition under the avatar model dir."""
    root = avatar_model_dir(settings)
    if not root.exists():
        return None
    matches = sorted(root.rglob("*.model3.json"))
    return matches[0] if matches else None


def expression_files(settings: Settings) -> dict[str, str]:
    """name -> served URL for every *.exp3.json next to the model."""
    model = find_model3(settings)
    if model is None:
        return {}
    root = avatar_model_dir(settings)
    out: dict[str, str] = {}
    for exp in sorted(model.parent.glob("*.exp3.json")):
        name = exp.name.removesuffix(".exp3.json")
        out[name] = f"/avatar-model/{exp.relative_to(root).as_posix()}"
    return out


def expression_names(settings: Settings) -> list[str]:
    return list(expression_files(settings))


def avatar_info(settings: Settings) -> dict:
    """What the frontend should render. Computed per request so dropping a
    model in and refreshing the page is enough — no restart."""
    model = find_model3(settings)
    if model is not None and vendor_ready(settings):
        relative = model.relative_to(avatar_model_dir(settings)).as_posix()
        param_map: dict = {}
        map_file = model.parent / "amadeus.map.json"
        if map_file.exists():
            import json

            try:
                param_map = json.loads(map_file.read_text())
            except ValueError:
                param_map = {}
        return {
            "renderer": "live2d",
            "model_url": f"/avatar-model/{relative}",
            "scale": settings.avatar_scale,
            "offset_y": settings.avatar_offset_y,
            "param_map": param_map,
            "expressions": expression_files(settings),
        }
    if model is not None and not vendor_ready(settings):
        return {
            "renderer": "none",
            "note": "Live2D model found but runtime missing — run: python -m amadeus avatar-setup",
        }
    return {
        "renderer": "none",
        "note": "No avatar model installed. Run `python -m amadeus avatar-setup`, then "
        "place a Live2D model folder (containing a *.model3.json) in "
        f"{avatar_model_dir(settings)}",
    }


def download_avatar_vendor(settings: Settings) -> None:
    from .voice.providers import _fetch
    import httpx

    target = vendor_dir(settings)
    target.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=None) as client:
        for name, url in VENDOR_FILES.items():
            _fetch(client, url, target / name)
