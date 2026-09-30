# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from pydantic import ValidationError

from omni_infinity.demo.models import DemoRecord, DemoRequest
from omni_infinity.demo.schedule import PRESETS, segment_count
from omni_infinity.serve.models import JobStatus, Progress
from omni_infinity.serve.store import CorruptJob, JobNotFound, JobStore, _now


class DemoStore(JobStore):
    def create(self, request: DemoRequest) -> DemoRecord:
        with self._lock:
            demo_id = uuid.uuid4().hex
            self.job_dir(demo_id).mkdir()
            now = _now()
            total_steps = (
                segment_count(PRESETS[request.duration])
                * request.num_inference_steps
            )
            record = DemoRecord(
                id=demo_id,
                request=request,
                status=JobStatus.QUEUED,
                progress=Progress(completed_steps=0, total_steps=total_steps),
                created_at=now,
                updated_at=now,
            )
            self._write(record)
            return record

    def get(self, job_id: str) -> DemoRecord:
        with self._lock:
            path = self._record_path(job_id)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return DemoRecord.model_validate(payload)
            except FileNotFoundError:
                raise JobNotFound(f"demo {job_id!r} not found") from None
            except (json.JSONDecodeError, OSError, ValidationError) as exc:
                raise CorruptJob(f"demo {job_id!r} is corrupt: {exc}") from exc

    def _record_path(self, job_id: str) -> Path:
        try:
            return self.job_dir(job_id) / "demo.json"
        except ValueError:
            raise JobNotFound(f"demo {job_id!r} not found") from None

    def _write(self, record: DemoRecord) -> None:
        final = self._record_path(record.id)
        temporary = final.with_name("demo.json.tmp")
        payload = json.dumps(
            record.model_dump(mode="json"), sort_keys=True, indent=2
        )
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final)
