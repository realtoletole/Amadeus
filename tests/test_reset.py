from amadeus.cli import reset_memory


def test_reset_deletes_database(tmp_path, monkeypatch):
    monkeypatch.setenv("AMADEUS_DATA_DIR", str(tmp_path))
    db = tmp_path / "amadeus.db"
    db.write_text("fake data")
    assert reset_memory(assume_yes=True) is True
    assert not db.exists()


def test_reset_with_no_database_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("AMADEUS_DATA_DIR", str(tmp_path))
    assert reset_memory(assume_yes=True) is False


def test_reset_cancelled_without_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("AMADEUS_DATA_DIR", str(tmp_path))
    db = tmp_path / "amadeus.db"
    db.write_text("fake data")
    monkeypatch.setattr("builtins.input", lambda _: "no")
    assert reset_memory() is False
    assert db.exists()
