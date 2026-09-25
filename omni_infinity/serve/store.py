# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from omni_infinity.serve.models import (
    MAX_ERROR_LENGTH,
    TERMINAL_STATUSES,
    ArtifactMetadata,
    GenerationRequest,
    JobRecord,
    JobStatus,
    Progress,
)

logger = logging.getLogger(__name__)

ALLOWED_TRANSITIONS = {
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.CANCELLED},
    JobStatus.RUNNING: {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.SUCCEEDED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
}

_JOB_ID = re.compile(r"^[0-9a-f]{32}$")


class JobStoreError(RuntimeError):
    pass


class JobNotFound(JobStoreError):
    pass


class CorruptJob(JobStoreError):
    pass


class InvalidTransition(JobStoreError):
    pass


class InvalidProgress(JobStoreError, ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def create(self, request: GenerationRequest) -> JobRecord:
        with self._lock:
            job_id = uuid.uuid4().hex
            self.job_dir(job_id).mkdir()
            now = _now()
            record = JobRecord(
                id=job_id,
                request=request,
                status=JobStatus.QUEUED,
                progress=Progress(
                    completed_steps=0,
                    total_steps=request.num_inference_steps,
                ),
                created_at=now,
                updated_at=now,
            )
            self._write(record)
            return record

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            path = self._record_path(job_id)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return JobRecord.model_validate(payload)
            except FileNotFoundError:
                raise JobNotFound(f"job {job_id!r} not found") from None
            except (json.JSONDecodeError, OSError, ValidationError) as exc:
                raise CorruptJob(f"job {job_id!r} is corrupt: {exc}") from exc

    def transition(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: str | None = None,
        artifacts: ArtifactMetadata | None = None,
    ) -> JobRecord:
        with self._lock:
            record = self.get(job_id)
            if status not in ALLOWED_TRANSITIONS[record.status]:
                raise InvalidTransition(
                    f"cannot transition {record.status.value} to {status.value}"
                )
            now = _now()
            updates: dict = {"status": status, "updated_at": now}
            if status == JobStatus.RUNNING:
                updates["started_at"] = now
            if status in TERMINAL_STATUSES:
                updates["finished_at"] = now
            if status == JobStatus.SUCCEEDED:
                updates["progress"] = Progress(
                    completed_steps=record.progress.total_steps,
                    total_steps=record.progress.total_steps,
                )
                updates["artifacts"] = artifacts
            if status == JobStatus.FAILED:
                updates["error"] = (error or "job failed")[:MAX_ERROR_LENGTH]
            updated = record.model_copy(update=updates)
            self._write(updated)
            return updated

    def update_progress(
        self, job_id: str, completed: int, total: int
    ) -> JobRecord:
        with self._lock:
            record = self.get(job_id)
            if record.status != JobStatus.RUNNING:
                raise InvalidTransition(
                    "progress can only update a running job"
                )
            if total != record.progress.total_steps:
                raise InvalidProgress("progress total cannot change")
            if not 0 <= completed <= total:
                raise InvalidProgress(
                    "completed steps must be between zero and total"
                )
            updated = record.model_copy(
                update={
                    "progress": Progress(
                        completed_steps=completed, total_steps=total
                    ),
                    "updated_at": _now(),
                }
            )
            self._write(updated)
            return updated

    def recover_interrupted(self) -> list[str]:
        recovered = []
        with self._lock:
            for directory in sorted(self.root.iterdir()):
                if not directory.is_dir() or not _JOB_ID.fullmatch(
                    directory.name
                ):
                    continue
                try:
                    record = self.get(directory.name)
                except CorruptJob:
                    logger.exception(
                        "skipping corrupt job during recovery: %s",
                        directory.name,
                    )
                    continue
                if record.status == JobStatus.RUNNING:
                    self.transition(
                        record.id,
                        JobStatus.FAILED,
                        error="server restarted while job was running",
                    )
                    recovered.append(record.id)
        return recovered

    def job_dir(self, job_id: str) -> Path:
        self._validate_id(job_id)
        return self.root / job_id

    def input_path(self, job_id: str, name: str) -> Path:
        if name in {"", ".", ".."} or Path(name).name != name:
            raise ValueError("input name must be a filename")
        return self.job_dir(job_id) / name

    def video_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "output.mp4"

    def audio_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "output.wav"

    def _record_path(self, job_id: str) -> Path:
        try:
            return self.job_dir(job_id) / "job.json"
        except ValueError:
            raise JobNotFound(f"job {job_id!r} not found") from None

    def _write(self, record: JobRecord) -> None:
        final = self._record_path(record.id)
        temporary = final.with_name("job.json.tmp")
        payload = json.dumps(
            record.model_dump(mode="json"), sort_keys=True, indent=2
        )
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final)

    @staticmethod
    def _validate_id(job_id: str) -> None:
        if not _JOB_ID.fullmatch(job_id):
            raise ValueError(
                "job id must be 32 lowercase hexadecimal characters"
            )
