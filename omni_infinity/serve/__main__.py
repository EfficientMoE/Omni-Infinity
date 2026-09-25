# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import uvicorn

from omni_infinity.serve.app import ServerSettings


def main() -> None:
    settings = ServerSettings.from_env()
    if settings.workers != 1:
        raise ValueError("OMNI_WORKERS must be 1 for the job-serving API")
    uvicorn.run(
        "omni_infinity.serve.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
