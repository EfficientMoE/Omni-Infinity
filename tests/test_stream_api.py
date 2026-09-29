from __future__ import annotations

from fastapi.testclient import TestClient

from omni_infinity.serve.app import ServerSettings, create_app


def _body(**overrides):
    body = {
        "type": "fl2va",
        "prompt": "a red ball bouncing",
        "optimizations": [],
    }
    body.update(overrides)
    return body


def test_streams_are_absent_unless_enabled(tmp_path):
    settings = ServerSettings(jobs_dir=tmp_path, optimizations=())
    with TestClient(create_app(settings, lambda: object())) as client:
        assert client.post("/v1/streams", json=_body()).status_code == 404


def test_open_session_caps_and_reuses_job_errors(tmp_path):
    settings = ServerSettings(
        jobs_dir=tmp_path, optimizations=(), stream_enabled=True
    )
    with TestClient(create_app(settings, lambda: object())) as client:
        first = client.post("/v1/streams", json=_body())
        assert first.status_code == 202
        assert len(first.json()["stream_id"]) == 32

        second = client.post("/v1/streams", json=_body())
        assert second.status_code == 409
        assert second.json()["detail"] == "stream session limit reached"

        ref = client.post("/v1/streams", json=_body(type="ref2va"))
        assert ref.status_code == 501

        other = client.post(
            "/v1/streams", json=_body(model_arch="vdn-hybrid")
        )
        assert other.status_code == 409
        assert other.json()["detail"] == (
            "request profile does not match the loaded server profile"
        )
        assert len(list(tmp_path.iterdir())) == 1


def test_finish_releases_stream_session_slot(tmp_path):
    settings = ServerSettings(
        jobs_dir=tmp_path, optimizations=(), stream_enabled=True
    )
    with TestClient(create_app(settings, lambda: object())) as client:
        first = client.post("/v1/streams", json=_body())
        stream_id = first.json()["stream_id"]
        client.app.state.stream_service.finish(stream_id)

        second = client.post("/v1/streams", json=_body())
        assert second.status_code == 202


def test_stream_enabled_env_parses_bools(monkeypatch):
    monkeypatch.delenv("OMNI_STREAM_ENABLED", raising=False)
    monkeypatch.delenv("OMNI_STREAM_CHUNK_FRAMES", raising=False)
    assert ServerSettings.from_env().stream_enabled is False

    monkeypatch.setenv("OMNI_STREAM_ENABLED", "true")
    monkeypatch.setenv("OMNI_STREAM_CHUNK_FRAMES", "12")
    assert ServerSettings.from_env().stream_enabled is True
    assert ServerSettings.from_env().stream_chunk_frames == 12

    monkeypatch.setenv("OMNI_STREAM_ENABLED", "0")
    assert ServerSettings.from_env().stream_enabled is False
