# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import uvicorn

from omni_infinity.serve.app import ServerSettings


def main() -> None:
    settings = ServerSettings.from_env()
    if settings.workers != 1:
        raise ValueError("OMNI_WORKERS must be 1 for the job-serving API")
    if settings.role in {"encoder", "denoiser", "decoder"}:
        raise NotImplementedError(
            "per-role servers are not wired up yet; use OMNI_ROLE=all "
            "or the single-host coordinator OMNI_ROLE=pipeline"
        )
    uvicorn.run(
        "omni_infinity.serve.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
