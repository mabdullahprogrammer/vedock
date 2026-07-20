from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import click
import requests


APP_NAME = os.getenv("APP_NAME", "Vedock")
CLI_NAME = os.getenv("CLI_NAME", "vedock")
DEFAULT_API = os.getenv("VEDOCK_API_URL", "https://vedock.ecorims.com/api/v1").rstrip("/")


def config_path() -> Path:
    configured = os.getenv("VEDOCK_CLI_CONFIG")
    return Path(configured) if configured else Path(os.getenv("APPDATA", Path.home())) / CLI_NAME / "config.json"


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {"api_url": DEFAULT_API}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"api_url": DEFAULT_API}


def save_config(data: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8", newline="\n")
    temporary.replace(path)


class ApiError(click.ClickException):
    pass


class Client:
    def __init__(self) -> None:
        self.config = load_config()
        self.api_url = self.config.get("api_url", DEFAULT_API).rstrip("/")

    def request(self, method: str, path: str, *, raw: bool = False, **kwargs: Any) -> Any:
        headers = kwargs.pop("headers", {})
        token = self.config.get("token")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = requests.request(method, f"{self.api_url}/{path.lstrip('/')}", headers=headers, timeout=kwargs.pop("timeout", 180), **kwargs)
        except requests.RequestException as exc:
            raise ApiError(f"Could not reach {self.api_url}: {exc}") from exc
        if raw:
            if response.status_code >= 400:
                raise ApiError(f"HTTP {response.status_code}: {response.text[:500]}")
            return response
        try:
            payload = response.json()
        except requests.JSONDecodeError as exc:
            raise ApiError(f"The server returned non-JSON data (HTTP {response.status_code}).") from exc
        if response.status_code >= 400 or not payload.get("ok"):
            error = payload.get("error") or {}
            detail = error.get("details")
            suffix = f"\n{json.dumps(detail, indent=2)}" if detail else ""
            raise ApiError(f"{error.get('message', 'Request failed')}{suffix}")
        return payload.get("data")


def print_json(value: Any) -> None:
    click.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def print_rows(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    if not rows:
        click.echo("No records.")
        return
    widths = []
    for key, title in columns:
        widths.append(min(50, max(len(title), *(len(str(row.get(key, ""))) for row in rows))))
    click.secho("┌" + "┬".join("─" * (width + 2) for width in widths) + "┐", fg="blue")
    click.secho("│" + "│".join(f" {title.ljust(width)} " for (_, title), width in zip(columns, widths)) + "│", fg="bright_blue", bold=True)
    click.secho("├" + "┼".join("─" * (width + 2) for width in widths) + "┤", fg="blue")
    status_colors = {"completed": "green", "running": "bright_blue", "awaiting_device": "yellow", "claimed": "cyan", "failed": "red", "cancelled": "bright_black", "queued": "yellow"}
    for row in rows:
        cells = []
        for (key, _), width in zip(columns, widths):
            value = str(row.get(key, ""))[:width].ljust(width)
            cells.append(click.style(value, fg=status_colors.get(str(row.get(key, "")).lower())) if key in {"status", "stage"} else value)
        click.echo("│" + "│".join(f" {value} " for value in cells) + "│")
    click.secho("└" + "┴".join("─" * (width + 2) for width in widths) + "┘", fg="blue")


def device_identity(client: Client) -> tuple[str, str]:
    device_id = str(client.config.get("device_id") or "")
    if not device_id:
        device_id = str(uuid.uuid4())
        client.config["device_id"] = device_id
        save_config(client.config)
    name = str(client.config.get("device_name") or platform.node() or "Vedock device")
    return device_id, name


@click.group(name=CLI_NAME, help=f"{APP_NAME} — Build any AI. No code. Full control.")
@click.option("--api-url", envvar="VEDOCK_API_URL", help="Override the Vedock API base URL.")
@click.pass_context
def cli(context: click.Context, api_url: str | None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    client = Client()
    if api_url:
        client.api_url = api_url.rstrip("/")
    context.obj = client


@cli.command()
@click.pass_obj
def doctor(client: Client) -> None:
    """Check this device and the hosted service connection."""
    from vedock_cli.device import local_device_report

    report = local_device_report()
    connected = False
    try:
        client.request("GET", "/system/doctor")
        connected = True
    except ApiError:
        pass
    click.echo(f"{APP_NAME} doctor")
    click.echo(f"Hosted service: {'connected' if connected else 'unreachable'} ({client.api_url})")
    click.echo(f"Python: {report['python']} ({report['python_executable']})")
    click.echo(f"CUDA: {'available' if report['cuda_available'] else 'unavailable'}")
    for runtime in report["runtimes"]:
        suffix = "installed" if runtime["installed"] else "missing " + ", ".join(runtime["missing"])
        click.echo(f"  {'[ok]' if runtime['installed'] else '[--]'} {runtime['name']}: {suffix}")


@cli.command()
@click.option("--username", prompt=True)
@click.option("--password", prompt=True, hide_input=True)
@click.option("--token-name", default="CLI", show_default=True)
@click.pass_obj
def login(client: Client, username: str, password: str, token_name: str) -> None:
    """Authenticate and save a revocable local API token."""
    data = client.request("POST", "/auth/login", json={"username": username, "password": password, "token_name": token_name})
    config = load_config()
    config.update({"api_url": client.api_url, "token": data["token"], "username": data["user"]["username"]})
    save_config(config)
    client.config = config
    device_id, device_name = device_identity(client)
    client.request("POST", "/devices/connect", json={"device_id": device_id, "device_name": device_name, "details": {"platform": platform.platform()}})
    click.echo(f"Logged in to {APP_NAME} as {data['user']['username']}.")


@cli.command()
@click.pass_obj
def whoami(client: Client) -> None:
    """Show the authenticated API identity."""
    data = client.request("GET", "/whoami")
    click.echo(f"{data['username']} <{data['email']}>")


@cli.command()
@click.option("--wait-seconds", type=int, default=30, show_default=True, hidden=True)
@click.pass_obj
def ui(client: Client, wait_seconds: int) -> None:
    """Open the installed Vedock desktop application (never a browser tab)."""
    from vedock_cli.desktop import launch_desktop

    launch_desktop(client.api_url)


@cli.group()
def models() -> None:
    """List, inspect, and run models."""


@models.command("list")
@click.pass_obj
def models_list(client: Client) -> None:
    rows = client.request("GET", "/models")
    print_rows(rows, [("slug", "MODEL"), ("name", "NAME"), ("task_type", "TASK"), ("runtime", "RUNTIME")])


@models.command("info")
@click.argument("model")
@click.pass_obj
def models_info(client: Client, model: str) -> None:
    print_json(client.request("GET", f"/models/{model}"))


@models.command("add-local")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--name")
@click.pass_obj
def models_add_local(client: Client, path: Path, name: str | None) -> None:
    """Register a model folder from this device without uploading its weights."""
    from vedock_cli.resources import register_model

    device_id, device_name = device_identity(client)
    client.request("POST", "/devices/connect", json={"device_id": device_id, "device_name": device_name, "details": {"platform": platform.platform()}})
    print_json(register_model(client, device_id, str(path), name))


@models.command("fork")
@click.argument("model")
@click.pass_obj
def models_fork(client: Client, model: str) -> None:
    """Create an editable local fork without copying source weights."""
    print_json(client.request("POST", f"/models/{model}/fork"))


@models.command("edit")
@click.argument("model")
@click.option("--name")
@click.option("--description")
@click.option("--output-pattern", help="Pattern containing {prompt} and {response}.")
@click.option("--visibility", type=click.Choice(["public", "private"]), help="Choose whether other accounts can discover and remix this model.")
@click.pass_obj
def models_edit(client: Client, model: str, name: str | None, description: str | None, output_pattern: str | None, visibility: str | None) -> None:
    """Edit local model metadata and its saved input/output pattern."""
    payload = {key: value for key, value in {"name": name, "description": description, "output_pattern": output_pattern, "visibility": visibility}.items() if value is not None}
    print_json(client.request("PATCH", f"/models/{model}", json=payload))


@models.command("set-image")
@click.argument("model")
@click.argument("image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_obj
def models_set_image(client: Client, model: str, image: Path) -> None:
    """Set a PNG, JPEG, or WebP cover on an owned model."""
    with image.open("rb") as stream:
        print_json(client.request("POST", f"/models/{model}/cover", files={"image": (image.name, stream)}))


@models.command("remove")
@click.argument("model")
@click.option("--yes", is_flag=True, help="Confirm recoverable removal from the active workspace.")
@click.pass_obj
def models_remove(client: Client, model: str, yes: bool) -> None:
    """Remove a model from the active workspace without deleting source files."""
    if not yes and not click.confirm(f"Remove {model} from the active workspace?"):
        return
    print_json(client.request("DELETE", f"/models/{model}"))


@models.command("use")
@click.argument("model")
@click.option("--prompt", help="Prompt text; asks interactively when omitted.")
@click.option("--temperature", type=float, default=0.9, show_default=True)
@click.option("--top-p", type=float, default=0.9, show_default=True)
@click.option("--top-k", type=int, default=50, show_default=True)
@click.option("--max-new-tokens", type=int, default=120, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--output-pattern", help="Override the saved pattern; must contain {prompt} and {response}.")
@click.option("--stop", "stop_sequences", multiple=True, help="Stop generation when this text is produced; repeatable.")
@click.option("--save", is_flag=True, help="Save the generation as a conversation.")
@click.pass_obj
def models_use(client: Client, model: str, prompt: str | None, temperature: float, top_p: float, top_k: int, max_new_tokens: int, seed: int, output_pattern: str | None, stop_sequences: tuple[str, ...], save: bool) -> None:
    prompt = prompt or click.prompt("Prompt")
    data = _run_inference(client, model, prompt, temperature, top_p, top_k, max_new_tokens, seed, output_pattern, stop_sequences, save)
    click.echo(data["text"])
    click.echo(f"\n[{data['device']} · {data['elapsed_seconds']}s]", err=True)


def _pairs(values: tuple[str, ...], option: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise click.UsageError(f"{option} values must use NAME=VALUE.")
        name, raw = value.split("=", 1)
        if not name.strip():
            raise click.UsageError(f"{option} needs a non-empty field name.")
        output[name.strip()] = raw
    return output


def _print_model_outputs(result: dict[str, Any]) -> None:
    for block in result.get("outputs") or []:
        kind = block.get("type")
        click.secho(str(block.get("label") or "Result"), fg="bright_blue", bold=True)
        if kind == "metric":
            click.secho(f"{block.get('value')} {block.get('unit') or ''}".rstrip(), fg="green", bold=True)
        elif kind == "probabilities":
            for item in block.get("items") or []:
                score = float(item.get("score") or 0)
                bar = "█" * min(30, max(0, int(round(score * 30))))
                click.echo(f"  {str(item.get('label')):<24} {score:>7.2%}  {click.style(bar, fg='blue')}")
        elif kind == "text":
            click.echo(str(block.get("value") or ""))
        elif kind == "table":
            columns = [str(item) for item in block.get("columns") or []]
            rows = [{columns[index]: value for index, value in enumerate(row)} for row in block.get("rows") or []]
            print_rows(rows, [(column, column.upper()) for column in columns])
        else:
            print_json(block.get("value", block))


@models.command("run")
@click.argument("model")
@click.option("--input", "input_values", multiple=True, help="Typed model input as NAME=VALUE; repeatable.")
@click.option("--file", "file_values", multiple=True, help="File/image input as NAME=PATH; repeatable.")
@click.option("--parameter", "parameter_values", multiple=True, help="Runtime parameter as NAME=VALUE; repeatable.")
@click.pass_obj
def models_run(client: Client, model: str, input_values: tuple[str, ...], file_values: tuple[str, ...], parameter_values: tuple[str, ...]) -> None:
    """Run any model through its declared typed input/output contract."""
    inputs = _pairs(input_values, "--input")
    paths = _pairs(file_values, "--file")
    parameters = _pairs(parameter_values, "--parameter")
    info = client.request("GET", f"/models/{model}")
    contract = (info.get("capabilities") or {}).get("runner") or {}
    for field in contract.get("inputs") or []:
        name = str(field["name"])
        if name in inputs or name in paths:
            continue
        if field.get("type") in {"image", "file"}:
            if field.get("required"):
                paths[name] = click.prompt(f"{field.get('label', name)} file")
        else:
            inputs[name] = click.prompt(str(field.get("label") or name), default=field.get("default"), show_default=field.get("default") is not None)
    if paths:
        from contextlib import ExitStack

        with ExitStack() as stack:
            files = {}
            for name, raw_path in paths.items():
                path = Path(raw_path).expanduser().resolve()
                if not path.is_file():
                    raise click.UsageError(f"Input file does not exist: {path}")
                files[name] = (path.name, stack.enter_context(path.open("rb")))
            result = client.request("POST", f"/models/{model}/run", files=files, data={**inputs, **parameters}, timeout=600)
    else:
        result = client.request("POST", f"/models/{model}/run", json={"inputs": inputs, "parameters": parameters}, timeout=600)
    _print_model_outputs(result)


def _run_inference(client: Client, model: str, prompt: str, temperature: float, top_p: float, top_k: int, max_new_tokens: int, seed: int, output_pattern: str | None, stop_sequences: tuple[str, ...], save: bool) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
    }
    if output_pattern is not None:
        parameters["output_pattern"] = output_pattern
    if stop_sequences:
        parameters["stop_sequences"] = list(stop_sequences)
    return client.request("POST", f"/models/{model}/infer", json={"prompt": prompt, "save_conversation": save, "parameters": parameters}, timeout=600)


def _stream_chat_message(client: Client, model: str, prompt: str, conversation_id: str | None, parameters: dict[str, Any], chat_settings: dict[str, Any]) -> str:
    response = client.request(
        "POST",
        f"/models/{model}/stream",
        raw=True,
        json={"prompt": prompt, "conversation_id": conversation_id, "parameters": parameters, **chat_settings},
        stream=True,
        timeout=600,
    )
    event_name = "message"
    saved_conversation = conversation_id
    for raw_line in response.iter_lines(decode_unicode=True):
        line = raw_line or ""
        if line.startswith("event:"):
            event_name = line[6:].strip()
            continue
        if not line.startswith("data:"):
            if not line:
                event_name = "message"
            continue
        try:
            payload = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if event_name == "message":
            click.echo(str(payload.get("token", "")), nl=False)
        elif event_name == "done":
            saved_conversation = payload.get("conversation_id") or saved_conversation
        elif event_name == "error":
            raise ApiError(str(payload.get("message") or "Generation failed"))
    click.echo()
    if not saved_conversation:
        raise ApiError("The local node did not return a conversation ID.")
    return str(saved_conversation)


@cli.command()
@click.argument("model")
@click.option("--prompt", help="Run one prompt and exit instead of interactive chat.")
@click.option("--temperature", type=float, default=0.9, show_default=True)
@click.option("--top-p", type=float, default=0.9, show_default=True)
@click.option("--top-k", type=int, default=50, show_default=True)
@click.option("--max-new-tokens", type=int, default=120, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--output-pattern", help="Override the saved pattern; must contain {prompt} and {response}.")
@click.option("--stop", "stop_sequences", multiple=True, help="Stop generation when this text is produced; repeatable.")
@click.option("--history/--no-history", default=True, help="Include earlier messages from this chat in model context.")
@click.option("--context", "context_override", help="Replace history with explicit context; {prompt} inserts the new message.")
@click.option("--context-limit", type=int, default=16000, show_default=True)
@click.pass_obj
def chat(client: Client, model: str, prompt: str | None, temperature: float, top_p: float, top_k: int, max_new_tokens: int, seed: int, output_pattern: str | None, stop_sequences: tuple[str, ...], history: bool, context_override: str | None, context_limit: int) -> None:
    """Stream a persistent conversation with one loaded model."""
    parameters = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_new_tokens": max_new_tokens, "seed": seed, "streaming": True}
    if output_pattern is not None:
        parameters["output_pattern"] = output_pattern
    if stop_sequences:
        parameters["stop_sequences"] = list(stop_sequences)
    chat_settings: dict[str, Any] = {"use_history": history, "context_override": context_override or "", "context_limit": context_limit}
    if prompt:
        _stream_chat_message(client, model, prompt, None, parameters, chat_settings)
        return
    click.echo("The model is loaded lazily and reused. Commands: /new, /history on|off, /context TEXT|clear, /exit")
    conversation_id = None
    while True:
        value = click.prompt("you")
        command = value.strip()
        if command.lower() in {"/exit", "/quit"}:
            break
        if command.lower() == "/new":
            conversation_id = None
            click.echo("Started a new chat.")
            continue
        if command.lower().startswith("/history "):
            setting = command.split(None, 1)[1].strip().lower()
            if setting not in {"on", "off"}:
                click.echo("Use /history on or /history off.")
            else:
                chat_settings["use_history"] = setting == "on"
                click.echo(f"History context is {setting}.")
            continue
        if command.lower().startswith("/context "):
            setting = command.split(None, 1)[1]
            chat_settings["context_override"] = "" if setting.strip().lower() == "clear" else setting
            click.echo("Context override cleared." if not chat_settings["context_override"] else "Context override updated.")
            continue
        click.echo("model> ", nl=False)
        conversation_id = _stream_chat_message(client, model, value, conversation_id, parameters, chat_settings)


@models.command("classify-image")
@click.argument("model")
@click.argument("image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--top-results", type=int, default=5, show_default=True)
@click.pass_obj
def models_classify_image(client: Client, model: str, image: Path, top_results: int) -> None:
    """Classify a local image with a fitted image-classification model."""
    with image.open("rb") as stream:
        result = client.request(
            "POST",
            f"/models/{model}/classify-image",
            files={"image": (image.name, stream)},
            data={"top_k": str(top_results)},
            timeout=600,
        )
    print_json(result)


@cli.group()
def datasets() -> None:
    """Import, inspect, transform, and validate datasets."""


@datasets.command("list")
@click.pass_obj
def datasets_list(client: Client) -> None:
    rows = client.request("GET", "/datasets")
    print_rows(rows, [("id", "ID"), ("name", "NAME"), ("file_format", "FORMAT"), ("row_count", "ROWS"), ("inspection_status", "STATUS")])


@datasets.command("community")
@click.argument("dataset", required=False)
@click.option("--import", "should_import", is_flag=True, help="Copy the selected starter into immutable local storage.")
@click.pass_obj
def datasets_community(client: Client, dataset: str | None, should_import: bool) -> None:
    """List local starter datasets or import one by ID."""
    if should_import:
        if not dataset:
            raise click.ClickException("Pass a community dataset ID with --import.")
        print_json(client.request("POST", f"/community-datasets/{dataset}/import"))
        return
    rows = client.request("GET", "/community-datasets")
    print_rows(rows, [("id", "ID"), ("name", "NAME"), ("description", "DESCRIPTION")])


@datasets.command("inspect")
@click.argument("path_or_url")
@click.option("--name")
@click.pass_obj
def datasets_inspect(client: Client, path_or_url: str, name: str | None) -> None:
    if path_or_url.startswith(("http://", "https://")):
        data = client.request("POST", "/datasets/import", json={"url": path_or_url, "name": name})
    else:
        path = Path(path_or_url).expanduser().resolve()
        if not path.is_file():
            raise click.ClickException(f"File not found: {path}")
        from vedock_cli.resources import inspect_dataset_file

        data = inspect_dataset_file(str(path))
    print_json(data)


@datasets.command("add-local")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--schema", type=click.Choice(["auto", "text_completion", "prompt_response", "instruction", "chat", "classification", "image_classification", "tabular_supervised"]), default="auto", show_default=True)
@click.pass_obj
def datasets_add_local(client: Client, path: Path, schema: str) -> None:
    """Prepare an immutable dataset version locally and register metadata only."""
    from vedock_cli.resources import register_dataset

    device_id, device_name = device_identity(client)
    client.request("POST", "/devices/connect", json={"device_id": device_id, "device_name": device_name, "details": {"platform": platform.platform()}})
    print_json(register_dataset(client, device_id, str(path), schema))


@datasets.command("validate")
@click.argument("dataset_version")
@click.pass_obj
def datasets_validate(client: Client, dataset_version: str) -> None:
    print_json(client.request("POST", f"/dataset-versions/{dataset_version}/validate"))


@datasets.command("transform")
@click.argument("dataset")
@click.option("--schema", "output_schema", type=click.Choice(["prompt_response", "text_completion", "instruction", "chat", "classification", "image_classification"]), default="prompt_response")
@click.option("--prompt-field", default="prompt")
@click.option("--response-field", default="story")
@click.option("--text-field", default="text")
@click.option("--instruction-field", default="instruction")
@click.option("--input-field", default="input")
@click.option("--output-field", default="output")
@click.option("--label-field", default="label")
@click.option("--image-field", default="image")
@click.option("--trim/--no-trim", default=True)
@click.option("--deduplicate/--keep-duplicates", default=True)
@click.option("--limit", type=int, default=0)
@click.pass_obj
def datasets_transform(client: Client, dataset: str, output_schema: str, prompt_field: str, response_field: str, text_field: str, instruction_field: str, input_field: str, output_field: str, label_field: str, image_field: str, trim: bool, deduplicate: bool, limit: int) -> None:
    if output_schema in {"prompt_response", "chat"}:
        mapping = {"prompt": prompt_field, "response": response_field}
    elif output_schema == "instruction":
        mapping = {"instruction": instruction_field, "input": input_field, "output": output_field}
    elif output_schema == "image_classification":
        mapping = {"image": image_field, "label": label_field}
    elif output_schema == "classification":
        mapping = {"text": text_field, "label": label_field}
    else:
        mapping = {"text": text_field}
    mapped_fields = list(mapping.values())
    operations = []
    if trim:
        operations.append({"type": "trim_whitespace", "config": {"fields": mapped_fields}})
    operations.append({"type": "remove_empty_records", "config": {"fields": mapped_fields}})
    if deduplicate:
        operations.append({"type": "remove_duplicates", "config": {}})
    data = client.request("POST", f"/datasets/{dataset}/transform", json={"operations": operations, "output_schema": output_schema, "field_mapping": mapping, "limit_examples": limit})
    print_json(data)


@cli.command()
@click.argument("model")
@click.option("--dataset", required=True, help="Processed dataset version ID.")
@click.option("--method", type=click.Choice(["lora", "full"]), default="lora")
@click.option("--output-name", default="cli-trained-model")
@click.option("--epochs", type=float, default=1.0)
@click.option("--max-steps", type=int, default=1)
@click.option("--learning-rate", type=float, default=0.0002)
@click.option("--batch-size", type=int, default=1)
@click.pass_obj
def train(client: Client, model: str, dataset: str, method: str, output_name: str, epochs: float, max_steps: int, learning_rate: float, batch_size: int) -> None:
    parameters = {"output_model_name": output_name, "training_method": method, "num_train_epochs": epochs, "max_steps": max_steps, "learning_rate": learning_rate, "per_device_train_batch_size": batch_size}
    print_json(client.request("POST", f"/train/{model}", json={"dataset": dataset, "parameters": parameters}))


@cli.group()
def jobs() -> None:
    """Inspect and control background jobs."""


@jobs.command("list")
@click.pass_obj
def jobs_list(client: Client) -> None:
    rows = client.request("GET", "/jobs")
    print_rows(rows, [("id", "JOB"), ("type", "TYPE"), ("status", "STATUS"), ("stage", "STAGE"), ("progress", "%")])


@jobs.command("edit")
@click.argument("job_id")
@click.option("--set", "values", multiple=True, metavar="NAME=VALUE", help="Change a parameter before the task is claimed; repeatable.")
@click.pass_obj
def jobs_edit(client: Client, job_id: str, values: tuple[str, ...]) -> None:
    """Edit an unclaimed hosted training task."""
    parameters: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise click.ClickException(f"Use NAME=VALUE, not {item!r}.")
        name, raw = item.split("=", 1)
        try:
            parameters[name.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            parameters[name.strip()] = raw
    print_json(client.request("PATCH", f"/jobs/{job_id}", json={"parameters": parameters}))


def _local_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return bool(__import__("shutil").which("nvidia-smi"))


@jobs.command("run")
@click.argument("job_id")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", show_default=True)
@click.option("--precision", type=click.Choice(["float32", "float16", "bfloat16"]), default=None)
@click.option("--publish/--keep-local", default=None, help="Publish automatically or keep the finalized model local; otherwise ask.")
@click.option("--yes", is_flag=True, help="Confirm local execution without another prompt (used by the desktop app).")
@click.pass_obj
def jobs_run(client: Client, job_id: str, device: str, precision: str | None, publish: bool | None, yes: bool) -> None:
    """Claim one hosted task and execute it on this computer."""
    from vedock_cli.local_jobs import ensure_runtime, run_claimed_job

    device_id, device_name = device_identity(client)
    from vedock_cli.resources import sync_pending_requests

    sync_pending_requests(client, device_id)
    record = client.request("GET", f"/jobs/{job_id}")
    if record["status"] == "awaiting_device":
        selected_device = ("cuda" if _local_cuda_available() else "cpu") if device == "auto" else device
        selected_precision = precision or ("float16" if selected_device == "cuda" else "float32")
        client.request("PATCH", f"/jobs/{job_id}", json={"parameters": {"device": selected_device, "precision": selected_precision}})
    manifest = client.request("GET", f"/jobs/{job_id}/manifest", headers={"X-Vedock-Device": device_id})
    click.echo(f"Model:   {manifest['model']['name']}")
    click.echo(f"Dataset: {manifest['dataset']['name']} · {manifest['dataset']['rows']} rows")
    click.echo(f"Method:  {manifest['parameters'].get('training_method')}")
    if not yes and not click.confirm("Run this task on this computer?", default=True):
        click.echo("Nothing was claimed or started.")
        return
    # Runtime readiness is deliberately checked before the hosted task is
    # claimed. A failed or declined install can no longer strand a task in the
    # claimed state.
    try:
        ensure_runtime(str(manifest["runtime"]))
    except Exception:
        if record.get("status") == "claimed" and record.get("claimed_by_device") == device_id:
            try:
                client.request("POST", f"/jobs/{job_id}/release", json={"device_id": device_id, "reason": "Runtime readiness check failed"})
            except Exception:
                pass
        raise
    click.secho(f"Claiming {job_id} on {device_name}", fg="blue", bold=True)
    claimed = client.request("POST", f"/jobs/{job_id}/claim", json={"device_id": device_id, "device_name": device_name})
    manifest = claimed["manifest"]
    try:
        result = run_claimed_job(client, job_id, manifest, device_id, publish)
    except Exception:
        try:
            client.request("POST", f"/jobs/{job_id}/release", json={"device_id": device_id, "reason": "Local worker did not start successfully"})
        except Exception:
            pass
        raise
    if isinstance(result, dict) and result.get("published") is False:
        click.echo("Use the desktop app or rerun with --publish when you are ready.")


@cli.group(name="list")
def list_commands() -> None:
    """Friendly aliases for listing connected resources."""


@list_commands.command("jobs")
@click.pass_obj
def list_jobs_alias(client: Client) -> None:
    """List training tasks (alias for `vedock jobs list`)."""
    rows = client.request("GET", "/jobs")
    print_rows(rows, [("id", "JOB"), ("type", "TYPE"), ("status", "STATUS"), ("stage", "STAGE"), ("progress", "%")])


@jobs.command("show")
@click.argument("job_id")
@click.pass_obj
def jobs_show(client: Client, job_id: str) -> None:
    print_json(client.request("GET", f"/jobs/{job_id}"))


@jobs.command("logs")
@click.argument("job_id")
@click.option("--limit", default=500, type=int)
@click.pass_obj
def jobs_logs(client: Client, job_id: str, limit: int) -> None:
    logs = client.request("GET", f"/jobs/{job_id}/logs?limit={limit}")
    for entry in logs:
        click.echo(f"{entry.get('time', '')}  {entry.get('message', '')}")
        if entry.get("metrics"):
            click.echo(f"  {json.dumps(entry['metrics'])}")
        if entry.get("error"):
            click.echo(f"  ERROR: {entry['error']}")


@jobs.command("cancel")
@click.argument("job_id")
@click.pass_obj
def jobs_cancel(client: Client, job_id: str) -> None:
    print_json(client.request("POST", f"/jobs/{job_id}/cancel"))


@jobs.command("resume")
@click.argument("job_id")
@click.pass_obj
def jobs_resume(client: Client, job_id: str) -> None:
    """Return a failed or cancelled task to the queue without starting it."""
    print_json(client.request("POST", f"/jobs/{job_id}/resume"))


@jobs.command("delete")
@click.argument("job_id")
@click.option("--yes", is_flag=True)
@click.pass_obj
def jobs_delete(client: Client, job_id: str, yes: bool) -> None:
    """Delete a terminal task and its logs; finalized model artifacts are kept."""
    if not yes and not click.confirm(f"Delete task {job_id} and its logs?"):
        return
    print_json(client.request("DELETE", f"/jobs/{job_id}"))


@jobs.command("release")
@click.argument("job_id")
@click.pass_obj
def jobs_release(client: Client, job_id: str) -> None:
    """Release a claimed task that has not started running."""
    device_id, _ = device_identity(client)
    print_json(client.request("POST", f"/jobs/{job_id}/release", json={"device_id": device_id}))


@cli.command()
@click.argument("model_a")
@click.argument("model_b")
@click.option("--weight-a", type=float, default=0.7)
@click.option("--weight-b", type=float, default=0.3)
@click.option("--output-name", default="Merged model")
@click.option("--execute", is_flag=True, help="Execute only when every safety check passes.")
@click.pass_obj
def merge(client: Client, model_a: str, model_b: str, weight_a: float, weight_b: float, output_name: str, execute: bool) -> None:
    payload = {"model_a": model_a, "model_b": model_b, "weight_a": weight_a, "weight_b": weight_b, "output_name": output_name}
    path = "/merge" if execute else "/merge/compatibility"
    print_json(client.request("POST", path, json=payload, timeout=600))


@cli.group()
def versions() -> None:
    """List immutable model versions."""


@versions.command("list")
@click.argument("model")
@click.pass_obj
def versions_list(client: Client, model: str) -> None:
    rows = client.request("GET", f"/versions/{model}")
    print_rows(rows, [("id", "VERSION"), ("version_number", "NO"), ("label", "LABEL"), ("status", "STATUS"), ("sha256", "HASH")])


@cli.command()
@click.argument("model")
@click.option("--output", type=click.Path(path_type=Path), help="Destination ZIP path.")
@click.pass_obj
def export(client: Client, model: str, output: Path | None) -> None:
    response = client.request("POST", f"/export/{model}", raw=True, timeout=600)
    destination = output or Path.cwd() / f"{model}.zip"
    if destination.exists():
        raise click.ClickException(f"Refusing to overwrite existing file: {destination}")
    destination.write_bytes(response.content)
    click.echo(str(destination.resolve()))


if __name__ == "__main__":
    cli()
