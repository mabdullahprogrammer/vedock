# Vedock Hosted Control Plane and Connected Compute

## Compute boundary

`https://vedock.ecorims.com` is the hosted Vedock control plane and public inference service. Browser-only users do not use WebGPU. Public chat inference runs on the Vedock host PC against finalized models stored under `E:\Vedock\storage`.

Training is different. The hosted server coordinates dataset/model metadata, projects, parameters, and task state, but private source artifacts remain on the owner's connected device. A hosted task begins in `awaiting_device`.

```text
Hosted web
  -> select or prepare a device-local dataset and model source
  -> save safe metadata, model project and training configuration
  -> create inert task: awaiting_device
  -> authenticated owner sees the same task in CLI/desktop
  -> owner explicitly starts it on a Windows/Linux computer
  -> local client resolves private `device://` inputs or downloads public bases
  -> local worker trains on that computer
  -> progress and logs stream to the hosted account
  -> owner reviews the finalized artifact
  -> only necessary inference/edit files are uploaded when published
```

The hosted server never launches an `awaiting_device` task. Merely opening the site, saving a project, restarting Vedock, or creating a task cannot start training.

When a user enters `D:\models\one` on the web, the hosted process never resolves that string against its own filesystem. It relays the locator only to the selected authenticated device. That client validates the local folder, stores the real path in its private configuration, returns a capability/hash manifest, and the control plane replaces the locator with an opaque `device://RESOURCE_ID` reference.

## Connected client

The Windows installer is a small one-file bootstrapper. It installs:

- the global `vedock` command in a user-owned bin directory added to user PATH;
- the native Vedock desktop controller;
- the fixed connection to `https://vedock.ecorims.com`;
- a private Python environment for the client.

It does not ask for a host or port. The installer detects an existing connected client and opens it without downloading it again. PyTorch, Transformers, PEFT/LoRA, image runtimes and fast-ML packages are installed only when the selected task requires them and are skipped when already present. CUDA is selected only after local hardware detection. Linux uses `install-vedock.sh` and registers `vedock` in `~/.local/bin` plus a desktop entry.

The hosted `/system` page and API deliberately redact the server's CPU, GPU, RAM, packages, Python executable, filesystem paths and protected directories. The Device and Runtimes pages in Vedock Desktop inspect only the computer on which the connected client is installed.

## Device task states

```text
awaiting_device -> claimed -> running -> awaiting_publish -> completed
              ^       |
              +-------+ release before worker start
                                   \-> failed
                                   \-> cancelled
failed/cancelled -> awaiting_device (manual resume; never auto-start)
```

Only the task owner can inspect, edit, claim, cancel or finalize a task. Parameters can be edited while it is waiting. The CLI previews the model, dataset, row count and method before claiming. A cancellation made on the web is observed by the local runner and stops the local worker without publishing a model.

## Artifact boundary

For device-local artifacts, the client reads its own immutable processed dataset and model folder directly and verifies the registered resource ownership/hash. No private file is uploaded for task preparation. For hosted/public bases, authenticated task-scoped downloads remain available; hashes are checked and ZIP extraction rejects path traversal.

Publication filters out checkpoints and unrelated local files. Accepted outputs are model weights/adapters, tokenizer assets, model/runtime configuration, model card and Vedock training metadata required for inference or continued editing. A finalized result may instead remain private on the device.

## Storage

The hosted deployment uses:

```text
E:\Vedock\
  instance\vedock.db
  storage\
    datasets\
    models\published\
    jobs\
    logos\
  distribution\
    VedockInstaller.exe
    VedockConnectedClient.zip
    install-vedock.sh
  logs\
```

The protected legacy StoryMaker project remains read-only. Verified model copies used by hosted inference are under `E:\Vedock\storage\models\published`.

## Scaling note

Public inference still consumes the host PC's RAM/VRAM. Before large public traffic, add a bounded inference queue, per-user limits, concurrent-model admission, idle unloading, storage quotas and operational backups. Connected training prevents other users' training workloads from consuming the host, but it does not remove the need to protect public inference.
