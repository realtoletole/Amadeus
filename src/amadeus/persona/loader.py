"""Load the persona from an editable markdown file.

The character is data, not code: ``<data_dir>/persona.md`` defines who the
companion is. On first run the file is seeded with the built-in Amadeus
persona as a starting template; from then on the file wins. Delete it to
regenerate the template.

Format:

    # Name

    Description paragraphs (become the persona description).

    ## Style
    - one rule per bullet
"""

from __future__ import annotations

from pathlib import Path

from ..config import Settings
from .profile import DEFAULT_PERSONA, Persona


def persona_path(settings: Settings) -> Path:
    return settings.data_dir / "persona.md"


def _render_template(persona: Persona) -> str:
    rules = "\n".join(f"- {rule}" for rule in persona.style_rules)
    return f"# {persona.name}\n\n{persona.description}\n\n## Style\n{rules}\n"


def _parse(text: str) -> Persona | None:
    lines = text.splitlines()
    name = ""
    description: list[str] = []
    style_rules: list[str] = []
    section = "description"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not name:
            name = stripped[2:].strip()
            continue
        if stripped.lower().startswith("## style"):
            section = "style"
            continue
        if section == "style":
            if stripped.startswith("- "):
                style_rules.append(stripped[2:].strip())
        else:
            description.append(line)
    body = "\n".join(description).strip()
    if not name or not body:
        return None
    return Persona(name=name, description=body, style_rules=style_rules)


def load_persona(settings: Settings) -> Persona:
    """Read the persona file, seeding it on first run. Falls back to the
    built-in persona if the file is unreadable or malformed (a broken
    character file should never brick the app)."""
    path = persona_path(settings)
    if not path.exists():
        settings.ensure_dirs()
        path.write_text(_render_template(DEFAULT_PERSONA), encoding="utf-8")
        return DEFAULT_PERSONA
    try:
        parsed = _parse(path.read_text(encoding="utf-8"))
    except OSError:
        parsed = None
    if parsed is None:
        print(f"[persona] {path} is empty or malformed — using the built-in persona")
        return DEFAULT_PERSONA
    return parsed
