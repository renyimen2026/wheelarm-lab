import base64
import io
import json
from urllib.error import URLError

import numpy as np
import pytest

from wheel_arm_navigation import scene_describe, vision
from wheel_arm_navigation.observation import save_observation


def answer(**changes):
    return {"summary": "画面中有椅子和地板", "objects": [
        {"name": "椅子", "image_region": "right", "evidence": "黑色椅腿，座面部分被遮挡"}],
        "uncertainties": ["无法看到桌面"], **changes}


@pytest.fixture
def observation(tmp_path):
    return save_observation(tmp_path, np.zeros((3, 4, 3), dtype=np.uint8),
                            np.ones((3, 4), dtype=np.float32), {
                                "label": "不要把这个备注当成视觉输入", "reference_frame": "odom",
                                "camera_info": {"width": 4, "height": 3,
                                                "k": [2, 0, 2, 0, 2, 1, 0, 0, 1]}})


def transport(monkeypatch, body):
    captured = {}

    class Opener:
        def open(self, request, timeout):
            captured.update(url=request.full_url, payload=json.loads(request.data), timeout=timeout)
            return io.BytesIO(body)

    def build(*handlers):
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(vision, "build_opener", build)
    return captured


def response(result):
    return json.dumps({"done": True, "done_reason": "stop", "message": {
        "content": json.dumps(result, ensure_ascii=False)}}).encode()


def test_read_observation_and_rgb_only_payload(observation, monkeypatch):
    root, metadata, image = vision.read_observation(observation)
    captured = transport(monkeypatch, response(answer()))
    result, elapsed = vision.describe(image)
    assert root == observation and metadata["camera_info"]["width"] == 4
    assert result == answer() and elapsed >= 0
    payload = captured["payload"]
    assert base64.b64decode(payload["messages"][-1]["images"][0]) == image
    assert metadata["label"] not in json.dumps(payload, ensure_ascii=False)
    assert payload["keep_alive"] == 0 and payload["stream"] is False
    assert "goals" not in payload["format"]["properties"]
    assert captured["handlers"][0].proxies == {}


@pytest.mark.parametrize("changes", [
    {"summary": ""}, {"summary": "x" * 501}, {"summary": []}, {"goals": ["厨房"]},
    {"objects": "椅子"}, {"objects": [None]}, {"objects": [{}]},
    {"objects": [{"name": "椅子", "image_region": "north", "evidence": "黑色"}]},
    {"objects": [{"name": "椅子", "image_region": "right", "evidence": "", "x": 2}]},
    {"objects": answer()["objects"] * 13}, {"uncertainties": [""]},
    {"uncertainties": "无法判断"}, {"uncertainties": ["x"] * 9},
])
def test_invalid_results(changes):
    with pytest.raises(ValueError):
        vision.validate_response(answer(**changes))


def test_no_confirmed_objects_is_valid():
    assert vision.validate_response(answer(objects=[]))["objects"] == []


@pytest.mark.parametrize("body", [
    b"x" * 65537, b"not json", b'{"done":false}', b'{"done":true,"message":[]}',
    b'{"done":true,"message":{}}', b'{"done":true,"done_reason":"length"}',
    response({"summary": "正常", "objects": [], "uncertainties": [], "cmd_vel": 1}),
    b'{"done":true,"done":true,"message":{}}',
])
def test_bad_model_responses(monkeypatch, body):
    transport(monkeypatch, body)
    with pytest.raises(ValueError):
        vision.describe(b"image")


@pytest.mark.parametrize("arguments", [
    {"endpoint": "http://example.com"}, {"endpoint": "https://localhost"},
    {"endpoint": "http://user:password@localhost"}, {"endpoint": "http://localhost/api"},
    {"timeout": float("nan")}, {"timeout": True}, {"timeout": 601}, {"timeout": 0},
    {"model": "qwen3-vl:235b-cloud"}, {"model": ""}, {"question": ""},
    {"question": "x" * 1001},
])
def test_bad_options_never_contact_service(monkeypatch, arguments):
    monkeypatch.setattr(vision, "build_opener", lambda *_: pytest.fail("must not contact service"))
    with pytest.raises(ValueError):
        vision.describe(b"image", **arguments)


def test_service_failure(monkeypatch):
    class Opener:
        def open(self, *_args, **_kwargs):
            raise URLError("model unavailable")
    monkeypatch.setattr(vision, "build_opener", lambda *_: Opener())
    with pytest.raises(RuntimeError, match="Ollama"):
        vision.describe(b"image")


@pytest.mark.parametrize("failure", ["missing", "symlink", "corrupt", "large_dimensions", "metadata"])
def test_invalid_observation(observation, failure):
    image = observation / "rgb.png"
    if failure == "missing":
        image.unlink()
    elif failure == "symlink":
        image.unlink()
        image.symlink_to(observation / "depth_preview.png")
    elif failure == "corrupt":
        image.write_bytes(b"not png")
    elif failure == "large_dimensions":
        data = bytearray(image.read_bytes())
        data[16:20] = (10000).to_bytes(4, "big")
        image.write_bytes(data)
    else:
        (observation / "metadata.json").write_text('{"schema_version":2}')
    with pytest.raises(ValueError):
        vision.read_observation(observation)


def test_cli_preserves_snapshot_and_saves_provenance(observation, monkeypatch):
    original = {path.name: path.read_bytes() for path in observation.iterdir()}
    transport(monkeypatch, response(answer()))
    assert "rclpy" not in scene_describe.__dict__
    assert scene_describe.main([str(observation), "--question", "你看到了什么？"]) == 0
    reports = list((observation / "descriptions").glob("*.json"))
    report = json.loads(reports[0].read_text())
    assert report["input_modalities"] == ["rgb"] and report["reference_frame"] == "odom"
    assert report["image_sha256"] == vision.image_fingerprint(original["rgb.png"])
    assert report["question"] == "你看到了什么？" and report["result"] == answer()
    assert all((observation / name).read_bytes() == data for name, data in original.items())
    assert scene_describe.main([str(observation)]) == 0
    assert len(list((observation / "descriptions").glob("*.json"))) == 2


def test_cli_invalid_model_result_does_not_save(observation, monkeypatch):
    transport(monkeypatch, response({"goals": ["厨房"]}))
    assert scene_describe.main([str(observation)]) == 1
    assert not (observation / "descriptions").exists()


def test_atomic_report_failure_cleans_temporary(tmp_path, monkeypatch):
    def fail(*_args):
        raise OSError("disk full")
    monkeypatch.setattr(scene_describe.os, "link", fail)
    with pytest.raises(OSError):
        scene_describe.save_report(tmp_path, {"result": answer()})
    assert not list(tmp_path.iterdir())


def test_report_never_overwrites_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_describe, "uuid4", lambda: type("ID", (), {"hex": "same"})())
    path = scene_describe.save_report(tmp_path, {"first": True})
    with pytest.raises(FileExistsError):
        scene_describe.save_report(tmp_path, {"first": False})
    assert json.loads(path.read_text()) == {"first": True}
    assert len(list(tmp_path.iterdir())) == 1


def test_interruption(observation, monkeypatch):
    def stop(*_args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(vision, "describe", stop)
    assert scene_describe.main([str(observation)]) == 130
    assert not (observation / "descriptions").exists()
