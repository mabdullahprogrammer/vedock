from __future__ import annotations

import argparse
import sys

from vedock import create_app
from vedock.services.jobs import run_job


def main() -> int:
    parser = argparse.ArgumentParser(description="Vedock controlled local worker")
    parser.add_argument("job_id")
    args = parser.parse_args()
    app = create_app(register_legacy=False)
    with app.app_context():
        return 0 if run_job(args.job_id) else 1


if __name__ == "__main__":
    sys.exit(main())
