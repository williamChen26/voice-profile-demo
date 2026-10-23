from datetime import datetime, timezone
from pathlib import Path
import json
import re
import uuid


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ProfileStore:
    def __init__(self, root):
        self.root = Path(root) / "profiles"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def create(self, name, payload):
        if not name or not name.strip() or len(name.strip()) > 80:
            raise ValueError("请输入 1～80 个字符的声纹名称")
        identifier = uuid.uuid4().hex
        profile = {**payload, "profile_id": identifier, "name": name.strip(), "created_at": timestamp()}
        write_json(self.root / f"{identifier}.json", profile)
        return profile

    def get(self, identifier):
        if not re.fullmatch(r"[0-9a-f]{32}", identifier or ""):
            raise ValueError("无效的 profile_id，请先录入并选择声纹")
        path = self.root / f"{identifier}.json"
        if not path.is_file():
            raise ValueError("声纹不存在")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self):
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.root.glob("*.json"))]

    def delete(self, identifier):
        self.get(identifier)
        (self.root / f"{identifier}.json").unlink()
