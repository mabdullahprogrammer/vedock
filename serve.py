from __future__ import annotations

from waitress import serve

from vedock import create_app


app = create_app()


if __name__ == "__main__":
    serve(
        app,
        host=app.config["APP_HOST"],
        port=app.config["APP_PORT"],
        threads=6,
        channel_timeout=600,
    )
