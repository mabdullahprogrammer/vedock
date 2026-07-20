# Running Vedock

## Hosted Vedock server

The current host is configured in `D:\LLM\vedock\.env` and listens on port 5464. Persistent data and finalized models are on `E:\Vedock`.

Production-style Windows start:

```powershell
Set-Location 'D:\LLM\vedock'
& '.\.venv\Scripts\python.exe' '.\serve.py'
```

Development start required by the product specification:

```powershell
Set-Location 'D:\LLM\vedock'
& '.\.venv\Scripts\python.exe' '.\run.py'
```

Open locally at `http://127.0.0.1:5464`. The public reverse proxy should forward `https://vedock.ecorims.com` to this service; reverse-proxy configuration is intentionally outside this project.

Do not run `worker.py` on the hosted server for user training. In `NODE_MODE=hosted_inference`, web training submissions are saved as `awaiting_device` tasks.

## Windows client

Download and run:

```text
https://vedock.ecorims.com/downloads/vedock-installer.exe
```

The installer has no port or host prompt. Choose an install folder and optionally create the desktop shortcut. Open a new terminal after installation:

```powershell
vedock login
vedock whoami
vedock doctor
vedock jobs list
vedock jobs show JOB_ID
vedock jobs edit JOB_ID --set learning_rate=0.0002 --set num_train_epochs=2
vedock jobs run JOB_ID
vedock jobs logs JOB_ID
vedock jobs cancel JOB_ID
vedock ui
```

`vedock list jobs` is a friendly alias for `vedock jobs list`. `vedock ui` opens the separate native desktop controller, not a browser.

## Linux client

```bash
curl -fsSLO https://vedock.ecorims.com/downloads/install-linux.sh
chmod +x install-linux.sh
./install-linux.sh
```

Ensure `~/.local/bin` is on PATH, then use the same `vedock` commands.

## Where training happens

- Clicking the web Train action creates a configuration/task on the hosted server.
- No computation starts there.
- `vedock jobs run JOB_ID`, or **Run on this computer** in the desktop app, claims the task.
- The selected user's computer downloads missing runtimes and performs training in its local Vedock workspace.
- Logs/status synchronize back to the hosted account.
- The final review can publish the filtered model artifact to `E:\Vedock\storage` or keep it only on the local device.

Browser-only users can chat without installing anything. That inference intentionally runs on the Vedock host PC, not in WebGPU.

## Verify the host without training

```powershell
Invoke-RestMethod 'http://127.0.0.1:5464/api/v1/'
Get-NetTCPConnection -LocalPort 5464 -State Listen
Set-Location 'D:\LLM\vedock'
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Expected API metadata includes `node_mode: hosted_inference` and `storage_location: E:\Vedock\storage`.
