from __future__ import annotations

import json
import os
import base64
import shutil
import subprocess
import sys
import threading
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any

import webview


APP_NAME = "Vedock"
CONTROL_PLANE = "https://vedock.ecorims.com"
DEFAULT_LOCATION = Path(os.getenv("LOCALAPPDATA", Path.home())) / APP_NAME
CLIENT_VERSION = "2026.07.20.6"


def _bundled_logo() -> str:
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    candidates = [bundle_root / "assets" / "logo.png"]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii") if path else ""


def _hidden_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def find_python() -> tuple[str, bool] | None:
    candidates = [
        shutil.which("py"),
        shutil.which("python"),
        str(Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python311" / "python.exe"),
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        launcher = Path(candidate).name.lower() == "py.exe"
        command = [candidate, "-3.11", "-c", "import sys;raise SystemExit(sys.version_info < (3,11))"] if launcher else [candidate, "-c", "import sys;raise SystemExit(sys.version_info < (3,11))"]
        if subprocess.run(command, capture_output=True, creationflags=_hidden_flags()).returncode == 0:
            return candidate, launcher
    return None


def python_command(runtime: tuple[str, bool], *arguments: str) -> list[str]:
    executable, launcher = runtime
    return [executable, "-3.11", *arguments] if launcher else [executable, *arguments]


class InstallerBridge:
    def __init__(self) -> None:
        self.window: Any = None
        self.installing = False
        self.installed_location: Path | None = None

    @staticmethod
    def _saved_config() -> dict[str, Any]:
        config = Path(os.getenv("APPDATA", Path.home())) / "vedock" / "config.json"
        try:
            value = json.loads(config.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _saved_location() -> Path:
        value = InstallerBridge._saved_config().get("install_location")
        return Path(value).expanduser().resolve() if value else DEFAULT_LOCATION

    def state(self) -> dict[str, Any]:
        location = self._saved_location()
        existing = self._installed_executable(location).is_file()
        current = existing and self._saved_config().get("client_version") == CLIENT_VERSION
        return {
            "location": str(location),
            "server": CONTROL_PLANE,
            "platform": "Windows 10 / 11",
            "installed": current,
            "update_available": existing and not current,
        }

    @staticmethod
    def _installed_executable(location: Path) -> Path:
        return location / "runtime" / "Scripts" / "vedock.exe"

    def check_installation(self, location: str) -> dict[str, Any]:
        destination = Path(location or DEFAULT_LOCATION).expanduser().resolve()
        existing = self._installed_executable(destination).is_file()
        installed = existing and self._saved_config().get("client_version") == CLIENT_VERSION
        if installed:
            self.installed_location = destination
        return {"installed": installed, "update_available": existing and not installed}

    def choose_folder(self) -> str:
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG, directory=str(DEFAULT_LOCATION.parent))
        if not result:
            return ""
        # pywebview backends return either one string or a list/tuple.  Indexing
        # a string was previously returning only its first character (for
        # example, "C"), which made Browse appear broken.
        return str(result[0] if isinstance(result, (list, tuple)) else result)

    def _emit(self, message: str, percent: int, level: str = "normal") -> None:
        self.window.evaluate_js(f"window.installProgress({json.dumps(message)}, {int(percent)}, {json.dumps(level)})")

    def _run(self, command: list[str], percent: int, message: str, cwd: Path | None = None) -> None:
        self._emit(message, percent)
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=_hidden_flags())
        assert process.stdout
        tail = []
        for line in process.stdout:
            if line.strip():
                tail.append(line.strip())
                tail = tail[-8:]
        if process.wait() != 0:
            raise RuntimeError("\n".join(tail) or f"Installation command failed with exit code {process.returncode}.")

    def _download_client(self, destination: Path) -> None:
        request = urllib.request.Request(
            f"{CONTROL_PLANE}/downloads/vedock-client.zip",
            headers={"User-Agent": "VedockInstaller/0.1 (+https://vedock.ecorims.com)", "Accept": "application/zip"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                "Vedock could not download the connected client from vedock.ecorims.com. "
                f"Please check your connection and try again. Details: {exc}"
            ) from exc

    def install(self, location: str, desktop_shortcut: bool) -> dict[str, Any]:
        if self.installing:
            return {"ok": False, "message": "Installation is already running."}
        destination = Path(location or DEFAULT_LOCATION).expanduser().resolve()
        self.installing = True
        threading.Thread(target=self._install, args=(destination, bool(desktop_shortcut)), daemon=True).start()
        return {"ok": True}

    def _install(self, destination: Path, desktop_shortcut: bool) -> None:
        try:
            destination.mkdir(parents=True, exist_ok=True)
            self._emit("Checking Python 3.11", 7)
            runtime = find_python()
            if not runtime:
                winget = shutil.which("winget")
                if not winget:
                    raise RuntimeError("Python 3.11 is missing and Windows Package Manager is unavailable.")
                self._run(
                    [winget, "install", "--id", "Python.Python.3.11", "-e", "--silent", "--accept-package-agreements", "--accept-source-agreements"],
                    12,
                    "Installing Python quietly",
                )
                runtime = find_python()
            if not runtime:
                raise RuntimeError("Python 3.11 could not be installed.")

            archive_path = destination / "vedock-client.download"
            self._emit("Downloading the connected Vedock client", 20)
            self._download_client(archive_path)
            source_root = destination / "client"
            source_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    target = (source_root / member.filename).resolve()
                    if source_root not in target.parents and target != source_root:
                        raise RuntimeError("The downloaded client contains an unsafe path.")
                archive.extractall(source_root)
            archive_path.unlink(missing_ok=True)
            project = source_root / "vedock-client"
            environment = destination / "runtime"
            self._run(python_command(runtime, "-m", "venv", str(environment)), 35, "Creating the private Vedock runtime")
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            self._run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "pip"], 46, "Preparing package installer")
            self._run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(project / "requirements-client.txt")], 58, "Installing CLI and desktop shell")
            self._run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", "-e", str(project)], 73, "Registering Vedock commands")

            bin_directory = destination / "bin"
            bin_directory.mkdir(parents=True, exist_ok=True)
            command = bin_directory / "vedock.cmd"
            command.write_text(f'@echo off\r\n"{environment / "Scripts/vedock.exe"}" %*\r\n', encoding="utf-8")
            config_root = Path(os.getenv("APPDATA", destination)) / "vedock"
            config_root.mkdir(parents=True, exist_ok=True)
            config = config_root / "config.json"
            try:
                configuration = json.loads(config.read_text(encoding="utf-8")) if config.is_file() else {}
            except (OSError, json.JSONDecodeError):
                configuration = {}
            configuration.update(
                {
                    "api_url": f"{CONTROL_PLANE}/api/v1",
                    "device_id": configuration.get("device_id") or str(uuid.uuid4()),
                    "device_name": configuration.get("device_name") or os.getenv("COMPUTERNAME", "Vedock device"),
                    "install_location": str(destination),
                    "client_version": CLIENT_VERSION,
                }
            )
            config.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
            self._add_to_path(bin_directory)
            if desktop_shortcut:
                self._create_shortcuts(command, project / "vedock_cli" / "assets" / "logo.ico")
            self.installed_location = destination
            self._emit("Vedock is ready", 100, "success")
            self.window.evaluate_js("window.installComplete()")
        except Exception as exc:
            self._emit(str(exc), 100, "error")
            self.window.evaluate_js("window.installFailed()")
        finally:
            self.installing = False

    @staticmethod
    def _add_to_path(directory: Path) -> None:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_SET_VALUE) as key:
            try:
                current, _ = winreg.QueryValueEx(key, "Path")
            except FileNotFoundError:
                current = ""
            entries = [item for item in str(current).split(";") if item]
            if str(directory).lower() not in {item.lower() for item in entries}:
                winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, ";".join(entries + [str(directory)]))
        # Tell Explorer and newly opened terminals about the user PATH update.
        try:
            import ctypes

            ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, None)
        except Exception:
            pass

    @staticmethod
    def _create_shortcuts(command: Path, logo: Path) -> None:
        desktop = Path(os.getenv("USERPROFILE", Path.home())) / "Desktop" / "Vedock.lnk"
        start_menu = Path(os.getenv("APPDATA", Path.home())) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Vedock.lnk"
        start_menu.parent.mkdir(parents=True, exist_ok=True)
        target = command.parent.parent / "runtime" / "Scripts" / "vedock.exe"
        arguments = "ui"
        escaped_target = str(target).replace("'", "''")
        escaped_working = str(command.parent.parent).replace("'", "''")
        escaped_icon = str(logo if logo.is_file() else target).replace("'", "''")
        script_parts = ["$w=New-Object -ComObject WScript.Shell;"]
        for shortcut in (desktop, start_menu):
            escaped_shortcut = str(shortcut).replace("'", "''")
            script_parts.extend(
                [
                    f"$s=$w.CreateShortcut('{escaped_shortcut}');",
                    f"$s.TargetPath='{escaped_target}';",
                    f"$s.Arguments='{arguments}';",
                    f"$s.WorkingDirectory='{escaped_working}';",
                    f"$s.IconLocation='{escaped_icon},0';",
                    "$s.Description='Vedock connected device';$s.Save();",
                ]
            )
        script = "".join(script_parts)
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, creationflags=_hidden_flags())

    def launch(self) -> None:
        location = self.installed_location or self._saved_location()
        executable = location / "runtime" / "Scripts" / "vedock.exe"
        if executable.is_file():
            subprocess.Popen([str(executable), "ui"], creationflags=_hidden_flags())
        self.window.destroy()


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{--blue:#1769ff;--ink:#111318;--muted:#6d7480;--line:#e3e6eb;--paper:#f7f8fa;font-family:Inter,'Segoe UI',sans-serif}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--paper);height:100vh;overflow:hidden}.layout{height:100vh;display:grid;grid-template-columns:310px 1fr}.visual{background:#0c0f15;position:relative;display:flex;flex-direction:column;justify-content:space-between;padding:38px;color:#fff;overflow:hidden}.visual:before,.visual:after{content:'';position:absolute;border:1px solid #1769ff66;transform:rotate(45deg)}.visual:before{width:320px;height:320px;left:-145px;top:130px}.visual:after{width:470px;height:470px;left:-220px;top:55px}.mark{display:flex;align-items:center;gap:10px;font-size:22px;font-weight:800;z-index:1}.mark img{width:42px;height:42px;object-fit:contain}.visual-copy{z-index:1}.visual h1{font-size:39px;line-height:.98;letter-spacing:-.05em;margin:10px 0}.visual p{color:#929aaa}.steps{display:grid;gap:11px;z-index:1}.steps span{font-size:11px;color:#778193}.steps b{display:inline-grid;place-items:center;width:22px;height:22px;margin-right:8px;border:1px solid #33405a;border-radius:50%;color:#82aaff}.content{padding:40px 44px 24px;display:grid;grid-template-rows:auto 1fr auto;min-width:0}.eyebrow{font-size:10px;color:var(--blue);font-weight:800;letter-spacing:.13em}.content h2{font-size:32px;margin:7px 0}.content>header p{color:var(--muted);margin:0}.card{align-self:center;background:white;border:1px solid var(--line);border-radius:12px;padding:23px;box-shadow:0 16px 50px #18284a0b}.included{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-bottom:20px}.included div{padding:12px;border:1px solid #edf0f4;border-radius:8px}.included b{display:block;font-size:12px}.included small{color:var(--muted)}label{display:grid;gap:6px;font-size:11px;font-weight:750}.path{display:grid;grid-template-columns:1fr auto;gap:7px}input{border:1px solid #d5d9e0;border-radius:8px;padding:11px;font:12px Consolas,monospace}button{font:inherit;cursor:pointer}.browse{border:1px solid #d5d9e0;background:#fff;border-radius:8px;padding:0 14px}.check{display:flex;align-items:center;gap:8px;margin-top:14px;font-weight:500}.progress-wrap{margin-top:20px}.progress{height:6px;background:#e9ecf1;border-radius:8px;overflow:hidden}.progress i{display:block;width:0;height:100%;background:linear-gradient(90deg,var(--blue),#7657ff);transition:.3s}.progress-row{display:flex;justify-content:space-between;margin-top:8px;color:var(--muted);font-size:11px}.footer{border-top:1px solid var(--line);padding-top:18px;display:flex;justify-content:space-between;align-items:center}.footer small{color:var(--muted)}.install{border:0;background:var(--blue);color:white;border-radius:8px;padding:11px 22px;font-weight:750;min-width:130px}.install:disabled{opacity:.55}.error{color:#b52d25!important}.success{color:#177b46!important}@media(max-width:720px){.layout{grid-template-columns:1fr}.visual{display:none}.content{padding:28px}}
</style></head><body><div class="layout"><aside class="visual"><div class="mark"><img src="__VEDOCK_LOGO__"><b>VEDOCK</b></div><div class="visual-copy"><span class="eyebrow">CONNECTED COMPUTE</span><h1>Build there.<br>Train here.</h1><p>Your tasks stay synchronized. Your hardware starts only when you choose.</p></div><div class="steps"><span><b>1</b>Install the connected client</span><span><b>2</b>Sign in to your Vedock account</span><span><b>3</b>Claim and run a training task</span></div></aside><main class="content"><header><span class="eyebrow">WINDOWS 10 / 11</span><h2>Install Vedock</h2><p>A small global CLI and desktop controller. ML runtimes download only when required.</p></header><section class="card"><div class="included"><div><b>Global `vedock` command</b><small>Use it from any terminal</small></div><div><b>Desktop controller</b><small>White, black and blue native UI</small></div><div><b>Hosted connection</b><small>vedock.ecorims.com</small></div><div><b>On-demand runtimes</b><small>PyTorch, LoRA and CUDA when needed</small></div></div><label>Install location<div class="path"><input id="installLocation"><button class="browse" onclick="chooseFolder()">Browse</button></div></label><label class="check"><input id="shortcut" type="checkbox" checked>Create desktop shortcut</label><div class="progress-wrap"><div class="progress"><i id="bar"></i></div><div class="progress-row"><span id="message">Ready to install</span><b id="percent">0%</b></div></div></section><footer class="footer"><small>No ports, server fields or developer setup.</small><button id="install" class="install" onclick="beginInstall()">Install Vedock</button></footer></main></div><script>
const byId=id=>document.getElementById(id);
const locationInput=()=>byId('installLocation');
const installButton=()=>byId('install');
const progressMessage=()=>byId('message');
const progressPercent=()=>byId('percent');
const progressBar=()=>byId('bar');
const shortcutInput=()=>byId('shortcut');
function setError(error){const text=error&&error.message?error.message:String(error||'Unexpected installer error');progressMessage().textContent=text;progressMessage().className='error';const button=installButton();button.disabled=false;button.textContent='Try again';button.onclick=beginInstall}
async function showInstalled(){const button=installButton();button.disabled=false;button.textContent='Open Vedock';progressMessage().textContent='Vedock is current — nothing will be downloaded';progressMessage().className='success';progressPercent().textContent='100%';progressBar().style.width='100%';button.onclick=()=>window.pywebview.api.launch()}
function showReady(state){const button=installButton();button.disabled=false;button.onclick=beginInstall;button.textContent=state.update_available?'Update Vedock':'Install Vedock';progressMessage().textContent=state.update_available?'A newer connected client is ready':'Ready to install';progressMessage().className='';progressPercent().textContent='0%';progressBar().style.width='0%'}
async function init(){try{const state=await window.pywebview.api.state();locationInput().value=state.location;state.installed?showInstalled():showReady(state)}catch(error){setError(error)}}
async function chooseFolder(){try{const value=await window.pywebview.api.choose_folder();if(!value)return;locationInput().value=value;const state=await window.pywebview.api.check_installation(value);state.installed?showInstalled():showReady(state)}catch(error){setError(error)}}
async function beginInstall(){try{const chosen=locationInput().value.trim();if(!chosen){setError('Choose an install location first.');return}const state=await window.pywebview.api.check_installation(chosen);if(state.installed){showInstalled();return}const button=installButton();button.disabled=true;button.textContent=state.update_available?'Updating…':'Installing…';const result=await window.pywebview.api.install(chosen,shortcutInput().checked);if(!result||result.ok===false)setError(result&&result.message?result.message:'The installer could not start.')}catch(error){setError(error)}}
window.installProgress=(text,value,level)=>{progressMessage().textContent=text;progressMessage().className=level;progressPercent().textContent=value+'%';progressBar().style.width=value+'%'};
window.installComplete=()=>showInstalled();
window.installFailed=()=>{const button=installButton();button.disabled=false;button.textContent='Try again';button.onclick=beginInstall};
window.addEventListener('pywebviewready',init);
</script></body></html>"""


def installer_html() -> str:
    return HTML.replace("__VEDOCK_LOGO__", _bundled_logo())


def main() -> None:
    bridge = InstallerBridge()
    window = webview.create_window("Install Vedock", html=installer_html(), js_api=bridge, width=980, height=650, min_size=(720, 540), background_color="#f7f8fa")
    bridge.window = window
    webview.start(debug=False)


if __name__ == "__main__":
    main()
