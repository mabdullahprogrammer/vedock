# Vedock Local Compute Node Architecture

## Product boundary

Vedock is not a hosted training service that runs every user's workloads on one server. The distributable application is a **local compute node**. Each user installs it on their own machine, and that node owns:

- Dataset bytes and immutable processed versions
- Downloaded and imported model weights
- Loaded runtime processes
- Inference and streaming
- Training and dataset workers
- Exports and local logs

The current Flask application is that local node. Its storage root is controlled by `VEDOCK_STORAGE_ROOT`; it defaults to the application's local `storage` folder.

## Shared command surface

```text
Local Web GUI / PWA
        |
        | HTTP + SSE on localhost
        v
Vedock local API (source of truth)
        ^
        | HTTP + SSE on localhost
        |
Vedock CLI
        |
        v
Runtime adapters -> local models / local workers / local datasets
```

The GUI and CLI use the same local API and runtime schemas. The GUI must not implement a second model engine, and it must not spawn shell commands to imitate the CLI. Both clients send the same validated commands to the local API. This preserves one behavior contract without the security and quoting problems of shelling out from Flask.

The chat interface already sends streaming inference through `/api/v1/models/{model}/stream`, the same local API family used by the CLI.

## Future public control plane

A public Vedock domain should contain only coordination features:

- Account authentication
- Device/node registration
- Encrypted command routing
- Small project metadata explicitly synchronized by the user
- Product updates and runtime-schema discovery

It must not receive dataset files, model weights, prompts, generated images, training batches, or checkpoints by default.

Each local node opens an authenticated outbound connection to the control plane. Commands include a node identifier, user authorization, nonce, expiry, and signature. The node validates the command, executes it locally, and returns progress/events. No inbound port or reverse proxy to the user's computer is required.

## Deployment modes

### Current: local GUI

The user runs `start-vedock.cmd` and opens `http://127.0.0.1:5464`. All storage and compute are local.

The authenticated Developer page exposes one signed-ready Windows installer executable. The small wizard downloads the application payload and lets the user select core, LLM, fast-ML, and developer components. The internal payload excludes virtual environments, caches, instance databases, datasets, conversations, model artifacts, and secrets. A receiving user receives an independent storage root.

### Packaged desktop/PWA

A desktop shell or installed PWA launches the same local node and renders its GUI. The Python runtime can be bundled, while models remain separately downloadable artifacts.

### Connected node

The public control plane authenticates the user and routes signed commands to the user's selected local node. The UI may be delivered by the public domain or rendered locally, but compute and artifacts remain on the node.

## Non-negotiable production rules

- Never execute user workloads on the public control-plane host unless a separate hosted-compute product is explicitly introduced.
- Never upload local artifacts without an explicit, scoped user action.
- Keep loaded models cached by node, model version, device, and precision; do not reload them for each chat message.
- Expose runtime capabilities and parameters through schemas shared by GUI and CLI.
- Use job workers for training and expensive transformations.
- Return structured errors and event logs; experimental operations must not crash the GUI.
