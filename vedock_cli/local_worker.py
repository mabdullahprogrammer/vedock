from __future__ import annotations

import argparse

from vedock import create_app
from vedock.services.jobs import run_job


def main() -> int:
    parser = argparse.ArgumentParser(description="Vedock owner-device training worker")
    parser.add_argument("job_id")
    parser.add_argument("--database", required=True)
    parser.add_argument("--storage", required=True)
    arguments = parser.parse_args()
    app = create_app(
        {
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{arguments.database}",
            "STORAGE_ROOT": arguments.storage,
            "NODE_MODE": "local_compute",
            "MODEL_TRAINING_ENABLED": True,
            "LAUNCH_JOBS": False,
            "OFFLINE_MODE": False,
            "PROTECTED_ROOTS": (),
        },
        register_legacy=False,
    )
    with app.app_context():
        return 0 if run_job(arguments.job_id) else 1


if __name__ == "__main__":
    raise SystemExit(main())
