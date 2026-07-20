from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


class DesktopBridge:
    def __init__(self, api_url: str) -> None:
        from vedock_cli.main import Client

        self.client = Client()
        self.client.api_url = api_url.rstrip("/")
        self.processes: dict[str, subprocess.Popen[Any]] = {}

    def bootstrap(self) -> dict[str, Any]:
        from vedock_cli.device import local_device_report
        from vedock_cli.main import device_identity

        device_id, device_name = device_identity(self.client)
        authenticated = False
        username = self.client.config.get("username")
        if self.client.config.get("token"):
            try:
                user = self.client.request("GET", "/whoami")
                authenticated, username = True, user["username"]
            except Exception:
                pass
        return {
            "authenticated": authenticated,
            "username": username,
            "device_id": device_id,
            "device_name": device_name,
            "api_url": self.client.api_url,
            "device": local_device_report(),
        }

    def login(self, username: str, password: str) -> dict[str, Any]:
        from vedock_cli.main import load_config, save_config

        data = self.client.request("POST", "/auth/login", json={"username": username, "password": password, "token_name": "Desktop app"})
        configuration = load_config()
        configuration.update({"api_url": self.client.api_url, "token": data["token"], "username": data["user"]["username"]})
        save_config(configuration)
        self.client.config = configuration
        return {"ok": True, "username": data["user"]["username"]}

    def jobs(self) -> list[dict[str, Any]]:
        from vedock_cli.main import device_identity

        device_id, _ = device_identity(self.client)
        for job_id, process in list(self.processes.items()):
            if process.poll() is not None:
                self.processes.pop(job_id, None)
        records = self.client.request("GET", "/jobs")
        for record in records:
            try:
                manifest = self.client.request("GET", f"/jobs/{record['id']}/manifest", headers={"X-Vedock-Device": device_id})
                record["model_name"] = manifest["model"]["name"]
                record["dataset_name"] = manifest["dataset"]["name"]
                record["dataset_rows"] = manifest["dataset"]["rows"]
                record["runtime"] = manifest["runtime"]
            except Exception:
                pass
            record["local_process_active"] = record["id"] in self.processes
        return records

    def models(self) -> list[dict[str, Any]]:
        return self.client.request("GET", "/models")

    def model(self, slug: str) -> dict[str, Any]:
        return self.client.request("GET", f"/models/{slug}")

    def choose_file(self) -> str:
        import webview

        selected = webview.windows[0].create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False)
        return str(selected[0]) if selected else ""

    def run_model(self, slug: str, inputs: dict[str, Any], parameters: dict[str, Any], file_paths: dict[str, str]) -> dict[str, Any]:
        if not file_paths:
            return self.client.request("POST", f"/models/{slug}/run", json={"inputs": inputs, "parameters": parameters}, timeout=600)
        from contextlib import ExitStack

        with ExitStack() as stack:
            files = {}
            for name, value in file_paths.items():
                path = Path(value).expanduser().resolve()
                if not path.is_file():
                    raise ValueError(f"Input file does not exist: {path}")
                files[name] = (path.name, stack.enter_context(path.open("rb")))
            return self.client.request("POST", f"/models/{slug}/run", data={**inputs, **parameters}, files=files, timeout=600)

    def datasets(self) -> list[dict[str, Any]]:
        return self.client.request("GET", "/datasets")

    def device(self) -> dict[str, Any]:
        from vedock_cli.device import local_device_report

        return local_device_report()

    def runtimes(self) -> list[dict[str, Any]]:
        from vedock_cli.device import runtime_report

        return runtime_report()

    def job(self, job_id: str) -> dict[str, Any]:
        data = self.client.request("GET", f"/jobs/{job_id}")
        data["logs"] = self.client.request("GET", f"/jobs/{job_id}/logs?limit=100")
        data["local_process_active"] = job_id in self.processes and self.processes[job_id].poll() is None
        log_path = Path(os.getenv("LOCALAPPDATA", Path.home())) / "Vedock" / "logs" / f"{job_id}.log"
        data["local_log"] = log_path.read_text(encoding="utf-8", errors="replace")[-12_000:] if log_path.is_file() else ""
        return data

    def edit_job(self, job_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return self.client.request("PATCH", f"/jobs/{job_id}", json={"parameters": parameters})

    def release_job(self, job_id: str) -> dict[str, Any]:
        from vedock_cli.main import device_identity

        device_id, _ = device_identity(self.client)
        return self.client.request("POST", f"/jobs/{job_id}/release", json={"device_id": device_id, "reason": "Released from Vedock Desktop"})

    def run_job(self, job_id: str, device: str, precision: str, publish: bool) -> dict[str, Any]:
        if job_id in self.processes and self.processes[job_id].poll() is None:
            return {"ok": False, "message": "This task is already running on this device."}
        command = [
            sys.executable,
            "-m",
            "vedock_cli.main",
            "jobs",
            "run",
            job_id,
            "--device",
            device,
            "--precision",
            precision,
            "--yes",
            "--publish" if publish else "--keep-local",
        ]
        environment = os.environ.copy()
        environment["VEDOCK_API_URL"] = self.client.api_url
        environment["VEDOCK_ASSUME_YES"] = "1"
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        log_root = Path(os.getenv("LOCALAPPDATA", Path.home())) / "Vedock" / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        output = (log_root / f"{job_id}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=environment, creationflags=flags)
        output.close()
        self.processes[job_id] = process
        return {"ok": True, "message": "Readiness checks started locally. The task is claimed only after its runtime is ready."}

    def infer(self, model: str, prompt: str) -> dict[str, Any]:
        return self.client.request(
            "POST",
            f"/models/{model}/infer",
            json={"prompt": prompt, "parameters": {"max_new_tokens": 160, "temperature": 0.8, "top_p": 0.95}},
            timeout=600,
        )


def _asset(name: str) -> Path | None:
    candidates = [Path(__file__).resolve().parent / "assets" / name]
    return next((path for path in candidates if path.is_file()), None)


def _logo_data() -> str:
    logo = _asset("logo.png")
    return "data:image/png;base64," + base64.b64encode(logo.read_bytes()).decode("ascii") if logo else ""


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="vedock-control-plane" content="https://vedock.ecorims.com"><style>
:root{--blue:#1464ff;--ink:#10131a;--muted:#697182;--line:#e5e9f0;--paper:#f5f7fb;--white:#fff;--nav:#0d111a;font-family:Inter,"Segoe UI",sans-serif}*{box-sizing:border-box}body{margin:0;height:100vh;overflow:hidden;background:var(--paper);color:var(--ink)}button,input,select,textarea{font:inherit}button{cursor:pointer}.shell{display:grid;grid-template-columns:242px 1fr;height:100vh}.side{background:var(--nav);color:white;padding:25px 17px;display:flex;flex-direction:column}.brand{display:flex;align-items:center;gap:12px;padding:0 8px 22px;border-bottom:1px solid #252c38}.brand img{width:40px;height:40px;object-fit:contain}.brand strong{display:block;font-size:19px}.brand small{color:#8791a3;font-size:9px;letter-spacing:.12em}.nav{display:grid;gap:5px;margin-top:22px}.nav button{border:0;background:transparent;color:#a5adbb;text-align:left;padding:12px 14px;border-radius:9px}.nav button:hover,.nav button.active{background:#1b2432;color:#fff}.connected{margin-top:auto;padding:18px 8px 0;border-top:1px solid #252c38;font-size:12px}.connected i{display:inline-block;width:8px;height:8px;border-radius:50%;background:#2ed27d;margin-right:7px}.connected small{display:block;color:#7f8999;margin-top:5px}.main{min-width:0;overflow:auto;padding:31px 35px}.head{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:24px}.eyebrow{font-size:10px;letter-spacing:.14em;color:var(--blue);font-weight:800}.head h1{font-size:32px;letter-spacing:-.04em;margin:5px 0}.head p,p{color:var(--muted)}.button{border:0;border-radius:9px;background:var(--blue);color:white;font-weight:750;padding:10px 16px}.button.secondary{background:#eaf1ff;color:var(--blue)}.button.danger{background:#fff0ee;color:#b52c22}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:13px}.card,.stat,.panel{background:white;border:1px solid var(--line);border-radius:13px}.card{padding:18px}.card h3{margin:5px 0}.card p{font-size:12px;line-height:1.55}.meta{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}.pill{font-size:10px;font-weight:750;padding:5px 8px;border-radius:20px;background:#edf3ff;color:#1959c3}.pill.good{background:#e7f8ef;color:#177442}.pill.warn{background:#fff4d9;color:#805a00}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}.stat{padding:17px}.stat small{color:var(--muted)}.stat strong{display:block;font-size:25px;margin-top:5px}.jobs{display:grid;gap:9px}.job{display:grid;grid-template-columns:minmax(220px,1fr) 150px 180px 58px;align-items:center;text-align:left;padding:15px 17px;background:white;border:1px solid var(--line);border-radius:11px}.job:hover{border-color:#8db0ff;box-shadow:0 8px 24px #1464ff0c}.job strong,.job small{display:block}.job small{color:var(--muted);margin-top:3px}.progress{height:5px;background:#e9edf3;border-radius:9px;overflow:hidden;margin-top:7px}.progress i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),#7357ff)}.empty{padding:70px 30px;text-align:center;background:white;border:1px dashed #cdd4de;border-radius:13px}.device-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}.device-grid .panel{padding:20px}.device-grid small{color:var(--muted);display:block}.device-grid strong{display:block;margin-top:6px}.login{height:100vh;display:grid;grid-template-columns:1.05fr .95fr;background:white}.login-art{background:#0d111a;display:grid;place-items:center;position:relative;overflow:hidden}.login-art:before,.login-art:after{content:"";position:absolute;border:1px solid #1464ff66;transform:rotate(45deg)}.login-art:before{width:360px;height:360px}.login-art:after{width:520px;height:520px}.login-art img{width:210px;z-index:1}.login-form{display:flex;flex-direction:column;justify-content:center;padding:12%;max-width:560px}.login-form h1{font-size:42px;letter-spacing:-.05em}.login-form label{display:grid;gap:6px;font-size:12px;font-weight:700;margin:8px 0}input,select,textarea{border:1px solid #d8dde6;background:white;border-radius:8px;padding:11px}.error{color:#bc2c22;font-size:12px}dialog{border:0;border-radius:14px;padding:0;width:min(760px,92vw);box-shadow:0 35px 100px #0005}dialog::backdrop{background:#0c101899;backdrop-filter:blur(5px)}.detail{padding:24px}.detail-head{display:flex;justify-content:space-between;gap:15px}.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:16px 0}.detail-grid div{background:#f5f7fa;padding:12px;border-radius:8px}.detail-grid small{display:block;color:var(--muted)}.logs{height:190px;overflow:auto;background:#0d111a;color:#c5cede;border-radius:8px;padding:13px;font:11px/1.55 Consolas,monospace;white-space:pre-wrap}.actions{display:flex;justify-content:flex-end;gap:8px;margin-top:15px}.runner{margin-top:14px;padding-top:14px;border-top:1px solid var(--line)}.runner textarea{width:100%;min-height:82px;resize:vertical}.runner-output{white-space:pre-wrap;max-height:220px;overflow:auto;padding:12px;background:#f6f8fb;border-radius:8px;margin-top:9px}.publish-choice{display:flex;gap:9px;align-items:flex-start;padding:12px;background:#edf3ff;border:1px solid #cbdcff;border-radius:8px;margin-bottom:14px}.publish-choice input{margin-top:2px}.muted{color:var(--muted)}@media(max-width:850px){.shell{grid-template-columns:78px 1fr}.brand span,.nav button span,.connected span{display:none}.main{padding:22px}.stats,.device-grid{grid-template-columns:1fr 1fr}.job{grid-template-columns:1fr 110px}.job>*:nth-child(n+3){display:none}}
.dynamic-fields{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:16px 0}.dynamic-fields>label{display:flex;flex-direction:column;gap:5px;padding:11px;background:#f7f9fc;border:1px solid var(--line);border-radius:8px}.dynamic-fields small{color:var(--muted);font-size:9px}.dynamic-fields input,.dynamic-fields select,.dynamic-fields textarea{width:100%}.runner-parameters{margin-top:12px;border-top:1px solid var(--line);padding-top:12px}.runner-parameters summary{font-weight:750;cursor:pointer}.file-choice{display:flex;align-items:center;gap:8px}.desktop-metric{display:flex;flex-direction:column;padding:18px;background:#eaf1ff;border-radius:9px}.desktop-metric strong{font-size:30px;color:#164fb5}.desktop-score{position:relative;display:grid;grid-template-columns:1fr 1.7fr auto;align-items:center;gap:8px;margin-top:8px;font-size:10px}.desktop-score i{height:7px;background:#2563eb;border-radius:5px}.desktop-gallery{display:grid;grid-template-columns:1fr 1fr;gap:8px}.desktop-gallery img{width:100%;border-radius:8px}@media(max-width:850px){.dynamic-fields{grid-template-columns:1fr}}
</style></head><body><div id="app"></div><script>
const logo="__LOGO__";let state={boot:null,jobs:[],models:[],datasets:[],page:"tasks"};const api=(name,...args)=>window.pywebview.api[name](...args);const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));const gb=n=>n?`${(n/1073741824).toFixed(1)} GiB`:"Unknown";
async function init(){state.boot=await api("bootstrap");state.boot.authenticated?await showPage("tasks"):renderLogin()}
function renderLogin(){app.innerHTML=`<section class="login"><div class="login-art"><img src="${logo}"></div><form class="login-form" onsubmit="login(event)"><span class="eyebrow">CONNECTED DEVICE</span><h1>Your models.<br>Your hardware.</h1><p>Sign in to synchronize tasks. Device information shown here always comes from this computer.</p><label>Username or email<input id="user" required></label><label>Password<input id="pass" type="password" required></label><p class="error" id="loginError"></p><button class="button">Connect this device</button></form></section>`}
async function login(e){e.preventDefault();try{await api("login",user.value,pass.value);state.boot=await api("bootstrap");await showPage("tasks")}catch(err){loginError.textContent=String(err)}}
function shell(title,sub,body,tool=""){let links=[["tasks","Training tasks"],["models","Models"],["datasets","Datasets"],["device","Device"],["runtimes","Runtimes"]];app.innerHTML=`<div class="shell"><aside class="side"><div class="brand"><img src="${logo}"><span><strong>Vedock</strong><small>CONNECTED DEVICE</small></span></div><nav class="nav">${links.map(([id,label])=>`<button class="${state.page===id?'active':''}" onclick="showPage('${id}')"><span>${label}</span></button>`).join("")}</nav><div class="connected"><span><i></i>${esc(state.boot.device_name)}<small>${esc(state.boot.username)} · connected</small></span></div></aside><main class="main"><header class="head"><div><span class="eyebrow">THIS COMPUTER</span><h1>${esc(title)}</h1><p>${esc(sub)}</p></div>${tool}</header>${body}</main></div><dialog id="dialog"></dialog>`}
async function showPage(page){state.page=page;if(page==="tasks"){state.jobs=await api("jobs");return renderTasks()}if(page==="models"){state.models=await api("models");return renderModels()}if(page==="datasets"){state.datasets=await api("datasets");return renderDatasets()}if(page==="device"){state.boot.device=await api("device");return renderDevice()}if(page==="runtimes"){state.boot.device.runtimes=await api("runtimes");return renderRuntimes()}}
function renderTasks(){let running=state.jobs.filter(j=>j.status==="running").length,waiting=state.jobs.filter(j=>j.status==="awaiting_device").length,claimed=state.jobs.filter(j=>j.status==="claimed").length,done=state.jobs.filter(j=>j.status==="completed").length;let body=`<section class="stats"><div class="stat"><small>Waiting</small><strong>${waiting}</strong></div><div class="stat"><small>Claimed</small><strong>${claimed}</strong></div><div class="stat"><small>Running</small><strong>${running}</strong></div><div class="stat"><small>Completed</small><strong>${done}</strong></div></section><section class="jobs">${state.jobs.length?state.jobs.map(j=>`<button class="job" onclick="openJob('${j.id}')"><span><strong>${esc(j.model_name||j.config?.parameters?.output_model_name||j.type)}</strong><small>${esc(j.dataset_name||"Dataset details available after manifest check")}</small></span><span class="pill ${j.status==='completed'?'good':j.status==='failed'?'warn':''}">${esc(j.status.replaceAll('_',' '))}</span><span><small>${esc(j.stage.replaceAll('_',' '))}</small><div class="progress"><i style="width:${j.progress}%"></i></div></span><b>${j.progress}%</b></button>`).join(""):`<div class="empty"><h2>No training tasks</h2><p>Create a model task on Vedock Web, then refresh this device.</p></div>`}</section>`;shell("Training tasks","Created on the web. Executed only after you choose Run on this computer.",body,`<button class="button secondary" onclick="showPage('tasks')">Refresh</button>`)}
function renderModels(){let body=state.models.length?`<section class="cards">${state.models.map(m=>`<article class="card"><span class="eyebrow">${esc(m.task_type.replaceAll('_',' '))}</span><h3>${esc(m.name)}</h3><p>${esc(m.description||"No description yet.")}</p><div class="meta"><span class="pill">${esc(m.runtime)}</span><span class="pill">${esc(m.visibility)}</span></div><button class="button secondary" style="margin-top:14px" onclick="openModel('${m.slug}')">Run model</button></article>`).join("")}</section>`:`<div class="empty"><h2>No accessible models</h2><p>Models you own and public community models appear here.</p></div>`;shell("Models","Each model opens the inputs and outputs declared by its own runtime.",body)}
function renderDatasets(){let body=state.datasets.length?`<section class="cards">${state.datasets.map(d=>{let rec=d.statistics?.recommendations||[];return `<article class="card"><span class="eyebrow">${esc(d.file_format)} · ${d.row_count??0} rows</span><h3>${esc(d.name)}</h3><p>${esc(d.original_filename)}</p><div class="meta"><span class="pill ${d.inspection_status==='completed'?'good':'warn'}">${esc(d.inspection_status)}</span><span class="pill">${d.versions?.length||0} versions</span></div>${rec.length?`<p><b>${rec.length} suggested improvements</b><br>${esc(rec.slice(0,2).map(x=>x.title).join(' · '))}</p>`:"<p>No automatic cleanup warning is recorded.</p>"}</article>`}).join("")}</section>`:`<div class="empty"><h2>No datasets</h2><p>Upload on Vedock Web. Inspection, cleanup suggestions and immutable versions will appear here.</p></div>`;shell("Datasets","Inspection summaries from your account—never server filesystem paths.",body)}
function renderDevice(){let d=state.boot.device,g=d.gpus||[];let body=`<section class="device-grid"><article class="panel"><small>Operating system</small><strong>${esc(d.platform)}</strong></article><article class="panel"><small>Processor</small><strong>${esc(d.processor)}</strong><span class="muted">${d.cpu_count||'?'} logical cores</span></article><article class="panel"><small>Memory</small><strong>${gb(d.ram_total_bytes)}</strong><span class="muted">${gb(d.ram_available_bytes)} available</span></article><article class="panel"><small>GPU / CUDA</small><strong>${d.cuda_available?esc(g.map(x=>x.name).join(', ')):"CUDA unavailable"}</strong><span class="muted">${esc(d.cuda_version||'CPU workloads remain available')}</span></article><article class="panel"><small>Disk</small><strong>${gb(d.disk_free_bytes)} free</strong><span class="muted">${gb(d.disk_total_bytes)} total</span></article><article class="panel"><small>Python</small><strong>${esc(d.python)}</strong><span class="muted">Private connected runtime</span></article></section>`;shell("Device","These specifications were read from this computer—not the hosted Vedock server.",body,`<button class="button secondary" onclick="showPage('device')">Scan again</button>`)}
function renderRuntimes(){let body=`<section class="cards">${state.boot.device.runtimes.map(r=>`<article class="card"><span class="eyebrow">${r.installed?'READY':'ON DEMAND'}</span><h3>${esc(r.name)}</h3><p>${r.installed?'Every required module is already installed. Vedock will not download it again.':'Missing: '+esc(r.missing.join(', '))+'. Vedock asks before installing when a matching task is started.'}</p><span class="pill ${r.installed?'good':'warn'}">${r.installed?'Installed':'Not installed'}</span></article>`).join("")}</section>`;shell("Runtimes","Installed tools are detected locally and are never downloaded twice.",body,`<button class="button secondary" onclick="showPage('runtimes')">Scan again</button>`)}
async function openJob(id){let j=await api("job",id),p=j.config?.parameters||{},ours=j.claimed_by_device===state.boot.device_id,runnable=j.status==="awaiting_device"||(j.status==="claimed"&&ours),editable=j.status==="awaiting_device";dialog.innerHTML=`<section class="detail"><div class="detail-head"><div><span class="eyebrow">${esc(j.id)}</span><h2>${esc(p.output_model_name||"Training task")}</h2></div><button class="button secondary" onclick="dialog.close()">Close</button></div><div class="detail-grid"><div><small>Status</small><strong>${esc(j.status.replaceAll('_',' '))}</strong></div><div><small>Method</small><strong>${esc(p.training_method||'—')}</strong></div><div><small>Device</small><select id="deviceChoice" ${editable?'':'disabled'}><option value="auto">Auto detect</option><option value="cuda">NVIDIA CUDA</option><option value="cpu">CPU</option></select></div><div><small>Precision</small><select id="precisionChoice" ${editable?'':'disabled'}><option>float32</option><option>float16</option><option>bfloat16</option></select></div></div>${runnable?`<label class="publish-choice"><input id="publishFinal" type="checkbox" checked><span><b>Publish after final review</b><small>Only necessary inference/edit files are uploaded.</small></span></label>`:""}<div class="logs">${esc(j.local_log||j.logs.map(x=>x.message).join('\n')||'No events yet.')}</div><div class="actions">${j.status==='claimed'&&ours&&!j.local_process_active?`<button class="button danger" onclick="releaseJob('${j.id}')">Release task</button>`:""}${runnable&&!j.local_process_active?`<button class="button" onclick="runJob('${j.id}')">${j.status==='claimed'?'Resume on this computer':'Run on this computer'}</button>`:""}</div></section>`;dialog.showModal()}
async function runJob(id){if(!confirm("Vedock will check required runtimes first, then claim and run this task locally. Continue?"))return;let result=await api("run_job",id,deviceChoice.value,precisionChoice.value,publishFinal.checked);alert(result.message);dialog.close();await showPage("tasks")}
async function releaseJob(id){if(!confirm("Release this task back to your queue?"))return;await api("release_job",id);dialog.close();await showPage("tasks")}
let runnerFiles={};
function inputControl(f){let common=`data-run-input="${esc(f.name)}"`;if(["image","file"].includes(f.type))return `<div class="file-choice"><button class="button secondary" type="button" onclick="pickRunnerFile('${esc(f.name)}')">Choose ${esc(f.label)}</button><small id="file_${esc(f.name)}">No file selected</small></div>`;if(f.type==="textarea"||f.type==="json")return `<textarea ${common} placeholder="${esc(f.placeholder||'')}">${esc(f.default||'')}</textarea>`;if(f.type==="boolean")return `<input ${common} type="checkbox" ${f.default?'checked':''}>`;if(f.choices?.length)return `<select ${common}>${f.choices.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('')}${f.allow_custom?'<option value="">Another / unknown value</option>':''}</select>`;return `<input ${common} type="${["number","integer"].includes(f.type)?'number':'text'}" step="any" value="${esc(f.default||'')}">`}
function parameterControl(f){if(f.type==="boolean")return `<label><span>${esc(f.label)}</span><input data-run-parameter="${esc(f.name)}" type="checkbox" ${f.default?'checked':''}></label>`;if(f.choices?.length)return `<label><span>${esc(f.label)}</span><select data-run-parameter="${esc(f.name)}">${f.choices.map(x=>`<option value="${esc(x)}" ${x===f.default?'selected':''}>${esc(x)}</option>`).join('')}</select></label>`;return `<label><span>${esc(f.label)}</span><input data-run-parameter="${esc(f.name)}" type="${["integer","float"].includes(f.type)?'number':'text'}" step="any" value="${esc(f.default??'')}"></label>`}
async function openModel(slug){runnerFiles={};let m=await api("model",slug),contract=m.capabilities?.runner||{};dialog.innerHTML=`<section class="detail"><div class="detail-head"><div><span class="eyebrow">${esc((contract.interaction||m.task_type).replaceAll('_',' '))}</span><h2>${esc(contract.title||m.name)}</h2><p>${esc(contract.description||m.description||'')}</p></div><button class="button secondary" onclick="dialog.close()">Close</button></div><div class="dynamic-fields">${(contract.inputs||[]).map(f=>`<label><b>${esc(f.label)}</b><small>${esc(f.description||'')}</small>${inputControl(f)}</label>`).join('')}</div><details class="runner-parameters"><summary>Model parameters</summary><div class="dynamic-fields">${(m.inference_parameters||[]).map(parameterControl).join('')}</div></details><div class="actions"><button class="button" onclick="runTypedModel('${slug}')">${esc(contract.submit_label||'Run model')}</button></div><div class="runner-output" id="modelOutput">The model output will appear here.</div></section>`;dialog.showModal()}
async function pickRunnerFile(name){let path=await api("choose_file");if(!path)return;runnerFiles[name]=path;document.getElementById(`file_${name}`).textContent=path.split(/[\\/]/).pop()}
function renderRunnerOutput(result){return (result.outputs||[]).map(b=>{if(b.type==="metric")return `<section class="desktop-metric"><small>${esc(b.label)}</small><strong>${esc(b.value)} ${esc(b.unit||'')}</strong></section>`;if(b.type==="probabilities")return `<section><b>${esc(b.label)}</b>${(b.items||[]).map(x=>`<div class="desktop-score"><span>${esc(x.label)}</span><i style="width:${Number(x.score)*100}%"></i><b>${(Number(x.score)*100).toFixed(2)}%</b></div>`).join('')}</section>`;if(b.type==="text")return `<section><b>${esc(b.label)}</b><p>${esc(b.value)}</p></section>`;if(["images","image"].includes(b.type))return `<section class="desktop-gallery">${(b.items||[b]).map(x=>`<img src="${esc(x.url||x)}">`).join('')}</section>`;return `<pre>${esc(JSON.stringify(b.value??b,null,2))}</pre>`}).join('')}
async function runTypedModel(slug){modelOutput.textContent="Running model…";try{let inputs={},parameters={};document.querySelectorAll('[data-run-input]').forEach(x=>inputs[x.dataset.runInput]=x.type==='checkbox'?x.checked:x.value);document.querySelectorAll('[data-run-parameter]').forEach(x=>parameters[x.dataset.runParameter]=x.type==='checkbox'?x.checked:x.value);let r=await api("run_model",slug,inputs,parameters,runnerFiles);modelOutput.innerHTML=renderRunnerOutput(r)}catch(e){modelOutput.textContent="Error: "+String(e)}}
window.addEventListener("pywebviewready",init);
</script></body></html>'''


def _page() -> str:
    return HTML.replace("__LOGO__", _logo_data())


def _set_windows_icon() -> None:
    if os.name != "nt":
        return
    icon = _asset("logo.ico")
    if not icon:
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        handle = user32.FindWindowW(None, "Vedock · Connected Device")
        loaded = user32.LoadImageW(None, str(icon), 1, 0, 0, 0x0010)
        if handle and loaded:
            user32.SendMessageW(handle, 0x0080, 0, loaded)
            user32.SendMessageW(handle, 0x0080, 1, loaded)
    except Exception:
        pass


def launch_desktop(api_url: str) -> None:
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError("The desktop component is not installed. Reopen the Vedock installer and choose Repair.") from exc
    bridge = DesktopBridge(api_url)
    webview.create_window("Vedock · Connected Device", html=_page(), js_api=bridge, width=1200, height=800, min_size=(780, 580), background_color="#f5f7fb")
    webview.start(_set_windows_icon, debug=False)
