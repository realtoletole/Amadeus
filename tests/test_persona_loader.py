import pathlib

from amadeus.config import Settings
from amadeus.persona import DEFAULT_PERSONA, load_persona, persona_path


def make_settings(tmp_path) -> Settings:
    return Settings(data_dir=pathlib.Path(tmp_path))


def test_first_run_seeds_template_and_uses_default(tmp_path):
    settings = make_settings(tmp_path)
    persona = load_persona(settings)
    assert persona.name == DEFAULT_PERSONA.name
    seeded = persona_path(settings).read_text(encoding="utf-8")
    assert seeded.startswith(f"# {DEFAULT_PERSONA.name}")
    assert "## Style" in seeded


def test_custom_file_defines_the_character(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    persona_path(settings).write_text(
        "# Kagari\n\nYou are Kagari, a quiet archivist who speaks in short "
        "sentences and loves cataloguing memories.\n\n## Style\n"
        "- Keep replies under three sentences.\n- Never use exclamation marks.\n",
        encoding="utf-8",
    )
    persona = load_persona(settings)
    assert persona.name == "Kagari"
    assert "archivist" in persona.description
    assert persona.style_rules == [
        "Keep replies under three sentences.",
        "Never use exclamation marks.",
    ]
    rendered = persona.render()
    assert "Kagari" in rendered and "exclamation" in rendered


def test_malformed_file_falls_back_to_default(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    persona_path(settings).write_text("   \n\n", encoding="utf-8")
    assert load_persona(settings).name == DEFAULT_PERSONA.name


def test_roundtrip_seeded_template_parses_back(tmp_path):
    settings = make_settings(tmp_path)
    load_persona(settings)                       # seeds
    persona = load_persona(settings)             # parses the seeded file
    assert persona.name == DEFAULT_PERSONA.name
    assert persona.style_rules == DEFAULT_PERSONA.style_rules
    assert persona.description.strip() == DEFAULT_PERSONA.description.strip()
