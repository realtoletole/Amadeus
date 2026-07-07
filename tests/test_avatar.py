import pathlib

from amadeus import avatar
from amadeus.config import Settings


def make_settings(tmp_path) -> Settings:
    return Settings(data_dir=pathlib.Path(tmp_path))


def install_fake_model(settings, name="chara"):
    model_dir = avatar.avatar_model_dir(settings) / name
    model_dir.mkdir(parents=True)
    (model_dir / f"{name}.model3.json").write_text("{}")
    return model_dir


def install_fake_vendor(settings):
    vendor = avatar.vendor_dir(settings)
    vendor.mkdir(parents=True)
    for name in avatar.VENDOR_FILES:
        (vendor / name).write_text("// stub")


def test_no_model_means_empty_stage_with_instructions(tmp_path):
    info = avatar.avatar_info(make_settings(tmp_path))
    assert info["renderer"] == "none"
    assert "avatar-setup" in info["note"]


def test_model_without_runtime_warns(tmp_path):
    settings = make_settings(tmp_path)
    install_fake_model(settings)
    info = avatar.avatar_info(settings)
    assert info["renderer"] == "none"
    assert "avatar-setup" in info["note"]


def test_model_with_runtime_serves_live2d(tmp_path):
    settings = make_settings(tmp_path)
    install_fake_model(settings)
    install_fake_vendor(settings)
    info = avatar.avatar_info(settings)
    assert info["renderer"] == "live2d"
    assert info["model_url"] == "/avatar-model/chara/chara.model3.json"
    assert info["scale"] == 1.0
    assert info["param_map"] == {}


def test_param_map_is_served_when_present(tmp_path):
    settings = make_settings(tmp_path)
    model_dir = install_fake_model(settings)
    install_fake_vendor(settings)
    (model_dir / "amadeus.map.json").write_text('{"blush": "ParamSwitch2"}')
    info = avatar.avatar_info(settings)
    assert info["param_map"] == {"blush": "ParamSwitch2"}


def test_api_avatar_endpoint_and_model_mount(tmp_path):
    from fastapi.testclient import TestClient

    from amadeus.cognition import CognitionEngine
    from amadeus.config import RetrievalWeights
    from amadeus.memory import ConversationArchive, MemoryRetriever, MemoryStore
    from amadeus.memory.embeddings import HashingEmbedder
    from amadeus.persona import DEFAULT_PERSONA
    from amadeus.server import create_app

    settings = make_settings(tmp_path / "data")
    install_fake_model(settings)
    install_fake_vendor(settings)

    store = MemoryStore(tmp_path / "a.db", HashingEmbedder(dim=64))
    engine = CognitionEngine(
        provider=None,
        store=store,
        retriever=MemoryRetriever(store, RetrievalWeights()),
        archive=ConversationArchive(tmp_path / "a.db"),
        persona=DEFAULT_PERSONA,
    )
    app = create_app(
        engine,
        avatar_info=lambda: avatar.avatar_info(settings),
        vendor_dir=avatar.vendor_dir(settings),
        avatar_model_dir=avatar.avatar_model_dir(settings),
    )
    client = TestClient(app)
    info = client.get("/api/avatar").json()
    assert info["renderer"] == "live2d"
    assert client.get(info["model_url"]).status_code == 200
    assert client.get("/vendor/pixi.min.js").status_code == 200


def test_expression_files_discovered_and_served(tmp_path):
    settings = make_settings(tmp_path)
    model_dir = install_fake_model(settings)
    install_fake_vendor(settings)
    (model_dir / "blush.exp3.json").write_text('{"Parameters": []}')
    (model_dir / "gloom.exp3.json").write_text('{"Parameters": []}')
    assert avatar.expression_names(settings) == ["blush", "gloom"]
    info = avatar.avatar_info(settings)
    assert info["expressions"]["blush"] == "/avatar-model/chara/blush.exp3.json"
