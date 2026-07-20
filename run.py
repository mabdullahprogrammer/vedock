from __future__ import annotations

from vedock import create_app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=app.config["APP_HOST"],
        port=app.config["APP_PORT"],
        debug=app.config["DEBUG"],
        threaded=True,
        use_reloader=False,
    )
