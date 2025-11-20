import os
import io
import json
import time
import hashlib
import secrets
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load env
load_dotenv()

app = FastAPI(title="Arabic Voice Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants from env with sane defaults
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:3000").rstrip("/")
TOKEN_TTL_DEFAULT = int(os.getenv("TOKEN_TTL_DEFAULT", "300"))  # seconds
READ_MAX_BYTES = int(os.getenv("READ_MAX_BYTES", "1048576"))  # 1MB
RUN_TIMEOUT_MS = int(os.getenv("RUN_TIMEOUT_MS", "8000"))
OUTPUT_MAX_BYTES = int(os.getenv("OUTPUT_MAX_BYTES", "1048576"))
SANDBOX_DIRS = [p.strip() for p in os.getenv("SANDBOX_DIRS", "/tmp,/var/tmp").split(",") if p.strip()]
RUN_WHITELIST = [c.strip() for c in os.getenv("RUN_WHITELIST", "echo,dir,ls").split(",") if c.strip()]

# Database import (Mongo helpers)
try:
    from database import db, create_document, get_documents
except Exception:
    db = None
    create_document = None
    get_documents = None

# Utility: collections
COLL_CHAT = "chat"
COLL_AUDIT = "auditlog"
COLL_TOKEN = "token"
COLL_PERMISSION = "permissionrequest"

# Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = Field(default="llama3.1")
    messages: List[ChatMessage]
    stream: bool = True
    temperature: Optional[float] = 0.7

class PermissionPreview(BaseModel):
    action: str = Field(..., description="e.g., fs.read or cmd.run")
    params: Dict[str, Any] = Field(default_factory=dict)
    scope: str = Field(..., description="scope token will grant, e.g., fs.read or cmd.run")
    ttl_seconds: Optional[int] = None

class ApproveRequest(BaseModel):
    request_id: str

# Token helpers

def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def issue_token(scope: str, meta: Dict[str, Any], ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    ttl = ttl_seconds or TOKEN_TTL_DEFAULT
    token_plain = secrets.token_urlsafe(24)
    token_hash = _hash_token(token_plain)
    doc = {
        "scope": scope,
        "meta": meta,
        "token_hash": token_hash,
        "issued_at": _now(),
        "expires_at": _now() + timedelta(seconds=ttl),
        "revoked": False,
    }
    db[COLL_TOKEN].insert_one(doc)
    return {"token": token_plain, "expires_at": doc["expires_at"].isoformat()}


def validate_token(token: str, required_scope: str) -> Dict[str, Any]:
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    token_hash = _hash_token(token)
    rec = db[COLL_TOKEN].find_one({"token_hash": token_hash, "revoked": False})
    if not rec:
        raise HTTPException(status_code=401, detail="Invalid token")
    if rec.get("expires_at") and rec["expires_at"] < _now():
        raise HTTPException(status_code=401, detail="Token expired")
    if rec.get("scope") != required_scope:
        raise HTTPException(status_code=403, detail="Insufficient scope")
    return rec

# Basic endpoints
@app.get("/")
def root():
    return {"message": "Arabic Voice Assistant API running"}

@app.get("/health")
def health():
    # Ollama check (best-effort)
    ollama_ok = False
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.5)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    db_ok = bool(db is not None)
    return {
        "backend": "ok",
        "ollama": "ok" if ollama_ok else "unreachable",
        "database": "ok" if db_ok else "not_configured",
    }

@app.get("/test")
def test_database():
    resp = {
        "backend": "ok",
        "database": "not_configured",
        "collections": []
    }
    try:
        if db is not None:
            resp["database"] = "ok"
            resp["collections"] = db.list_collection_names()
    except Exception as e:
        resp["database"] = f"error: {str(e)[:80]}"
    return resp

# Chat proxy with streaming to SSE-style text chunks
@app.post("/chat")
def chat_stream(req: ChatRequest):
    def generate():
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": req.model,
            "messages": [m.model_dump() for m in req.messages],
            "stream": True,
            "temperature": req.temperature,
        }
        accumulated = []
        started = _now().isoformat()
        try:
            with requests.post(url, json=payload, stream=True, timeout=300) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        delta = data.get("message", {}).get("content", "") or data.get("response", "")
                        if delta:
                            accumulated.append(delta)
                            yield f"data: {delta}\n\n"
                        if data.get("done"):
                            break
                    except Exception:
                        # pass through non-JSON
                        yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: [error] {str(e)}\n\n"
        # Log chat if db
        try:
            if db is not None and accumulated:
                db[COLL_CHAT].insert_one({
                    "started_at": started,
                    "ended_at": _now(),
                    "model": req.model,
                    "messages": [m.model_dump() for m in req.messages],
                    "response": "".join(accumulated)
                })
        except Exception:
            pass

    headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(generate(), headers=headers)

# Speech endpoints (browser-first). Provided for completeness.
@app.post("/speech/stt")
async def speech_to_text(engine: str = Form(default="browser"), file: Optional[UploadFile] = File(default=None)):
    if engine == "browser":
        return {"note": "Use browser Web Speech API for STT", "engine": engine}
    raise HTTPException(status_code=501, detail="STT engine not installed on server")

@app.post("/speech/tts")
async def text_to_speech(engine: str = Form(default="browser"), text: str = Form(...)):
    if engine == "browser":
        return {"note": "Use browser SpeechSynthesis API for TTS", "engine": engine}
    raise HTTPException(status_code=501, detail="TTS engine not installed on server")

# Permission preview -> approve (token issuance)
@app.post("/permissions/preview")
def permissions_preview(pre: PermissionPreview):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    summary = {
        "action": pre.action,
        "scope": pre.scope,
        "params": pre.params,
    }
    doc = {
        "summary": summary,
        "status": "pending",
        "created_at": _now(),
        "ttl": pre.ttl_seconds or TOKEN_TTL_DEFAULT,
    }
    res = db[COLL_PERMISSION].insert_one(doc)
    return {"request_id": str(res.inserted_id), "summary": summary}

@app.post("/permissions/approve")
def permissions_approve(body: ApproveRequest):
    from bson import ObjectId
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    rec = db[COLL_PERMISSION].find_one({"_id": ObjectId(body.request_id)})
    if not rec:
        raise HTTPException(status_code=404, detail="Request not found")
    if rec.get("status") != "pending":
        raise HTTPException(status_code=400, detail="Already processed")
    scope = rec["summary"]["scope"]
    token_data = issue_token(scope=scope, meta=rec["summary"], ttl_seconds=rec.get("ttl"))
    db[COLL_PERMISSION].update_one({"_id": rec["_id"]}, {"$set": {"status": "approved", "approved_at": _now()}})
    return token_data

# Helpers for allowlists

def _is_path_allowed(path: str) -> bool:
    try:
        abs_p = os.path.abspath(path)
        for base in SANDBOX_DIRS:
            base_abs = os.path.abspath(base)
            if abs_p.startswith(base_abs + os.sep) or abs_p == base_abs:
                return True
    except Exception:
        return False
    return False

# Protected FS read
@app.get("/fs/read")
def fs_read(path: str, authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1]
    validate_token(token, required_scope="fs.read")
    if not _is_path_allowed(path):
        raise HTTPException(status_code=403, detail="Path not allowed")
    try:
        with open(path, "rb") as f:
            data = f.read(READ_MAX_BYTES)
        # Audit
        try:
            if db is not None:
                db[COLL_AUDIT].insert_one({
                    "ts": _now(),
                    "action": "fs.read",
                    "path": path,
                    "bytes": len(data),
                })
        except Exception:
            pass
        return StreamingResponse(io.BytesIO(data), media_type="application/octet-stream")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Protected command run
class RunBody(BaseModel):
    command: str
    args: List[str] = []

@app.post("/cmd/run")
def cmd_run(body: RunBody, authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1]
    validate_token(token, required_scope="cmd.run")
    if body.command not in RUN_WHITELIST:
        raise HTTPException(status_code=403, detail="Command not allowed")
    try:
        completed = subprocess.run(
            [body.command] + body.args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=RUN_TIMEOUT_MS / 1000.0,
            check=False,
            text=True,
        )
        out = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
        out_bytes = out.encode()[:OUTPUT_MAX_BYTES]
        # Audit
        try:
            if db is not None:
                db[COLL_AUDIT].insert_one({
                    "ts": _now(),
                    "action": "cmd.run",
                    "command": body.command,
                    "args": body.args,
                    "rc": completed.returncode,
                    "out_bytes": len(out_bytes),
                })
        except Exception:
            pass
        return JSONResponse({"returncode": completed.returncode, "output": out_bytes.decode(errors="ignore")})
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Command timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# History endpoints
@app.get("/history/chats")
def history_chats(limit: int = 20):
    if db is None:
        return []
    items = db[COLL_CHAT].find().sort("_id", -1).limit(limit)
    return [
        {
            "model": it.get("model"),
            "messages": it.get("messages", []),
            "response": it.get("response", ""),
            "started_at": it.get("started_at"),
        }
        for it in items
    ]

@app.get("/history/audit")
def history_audit(limit: int = 50):
    if db is None:
        return []
    items = db[COLL_AUDIT].find().sort("_id", -1).limit(limit)
    return [
        {
            "ts": it.get("ts"),
            "action": it.get("action"),
            "path": it.get("path"),
            "bytes": it.get("bytes"),
            "command": it.get("command"),
            "args": it.get("args"),
            "rc": it.get("rc"),
        }
        for it in items
    ]

# Export helpers
EXCLUDE_DIR_NAMES = {"node_modules", "__pycache__", ".git", "dist", "build", "logs"}


def _zip_dir(base_dir: str, out_zip_path: str, arc_base: Optional[str] = None):
    import zipfile
    with zipfile.ZipFile(out_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(base_dir):
            # prune excluded dirs in-place
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES and not d.startswith('.')]
            for fname in filenames:
                if fname == ".DS_Store":
                    continue
                fpath = os.path.join(dirpath, fname)
                if "/logs/" in fpath:
                    continue
                if arc_base:
                    arcname = os.path.join(arc_base, os.path.relpath(fpath, base_dir))
                else:
                    arcname = os.path.relpath(fpath, base_dir)
                try:
                    zf.write(fpath, arcname)
                except Exception:
                    pass

def _zip_dir_into_zipfile(zf, base_dir: str, arc_prefix: str = ""):
    for dirpath, dirnames, filenames in os.walk(base_dir):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES and not d.startswith('.')]
        for fname in filenames:
            if fname == ".DS_Store":
                continue
            fpath = os.path.join(dirpath, fname)
            if "/logs/" in fpath:
                continue
            rel = os.path.relpath(fpath, base_dir)
            arcname = os.path.join(arc_prefix, rel) if arc_prefix else rel
            try:
                zf.write(fpath, arcname)
            except Exception:
                pass


def _zip_dir_to_bytes(base_dir: str, arc_base: Optional[str] = None) -> bytes:
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(base_dir):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES and not d.startswith('.')]
            for fname in filenames:
                if fname == ".DS_Store":
                    continue
                fpath = os.path.join(dirpath, fname)
                if "/logs/" in fpath:
                    continue
                if arc_base:
                    arcname = os.path.join(arc_base, os.path.relpath(fpath, base_dir))
                else:
                    arcname = os.path.relpath(fpath, base_dir)
                try:
                    zf.write(fpath, arcname)
                except Exception:
                    pass
    buf.seek(0)
    return buf.read()

# Export backend as zip
@app.get("/export/backend.zip")
async def export_backend():
    import tempfile
    backend_dir = os.getcwd()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=",backend.zip")
    tmp.close()
    _zip_dir(backend_dir, tmp.name)
    return FileResponse(tmp.name, media_type="application/zip", filename="backend.zip")

# Export frontend as zip (in-memory to avoid any temp-file edge cases)
@app.get("/export/frontend.zip")
async def export_frontend():
    # In this environment, backend and frontend run in separate services.
    # This endpoint remains for backward compatibility but will proxy to the frontend service when possible.
    fe_url = f"{FRONTEND_BASE_URL}/export/frontend.zip"
    try:
        with requests.get(fe_url, timeout=30, stream=True) as r:
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=r.text[:200])
            headers = {"Content-Disposition": "attachment; filename=frontend.zip"}
            return StreamingResponse(r.iter_content(chunk_size=65536), media_type="application/zip", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"frontend proxy error: {str(e)[:200]}")

# Export whole project as zip
@app.get("/export/project.zip")
async def export_project():
    import tempfile
    import zipfile

    # Create temp zip file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=",project.zip")
    tmp.close()

    with zipfile.ZipFile(tmp.name, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add backend files under backend/
        _zip_dir_into_zipfile(zf, os.getcwd(), arc_prefix="backend")

        # Try to fetch frontend zip and merge under frontend/
        fe_url = f"{FRONTEND_BASE_URL}/export/frontend.zip"
        try:
            with requests.get(fe_url, timeout=60) as r:
                if r.status_code == 200:
                    from zipfile import ZipFile
                    from io import BytesIO
                    zbytes = BytesIO(r.content)
                    with ZipFile(zbytes) as fz:
                        for info in fz.infolist():
                            if info.is_dir():
                                continue
                            # write into combined archive with prefix 'frontend/'
                            data = fz.read(info.filename)
                            arcname = os.path.join("frontend", info.filename)
                            zf.writestr(arcname, data)
        except Exception:
            # If frontend unreachable, still return backend-only zip
            pass

    return FileResponse(tmp.name, media_type="application/zip", filename="project.zip")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
