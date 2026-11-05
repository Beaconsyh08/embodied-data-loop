"""Issue or revoke independently scoped API keys; never print a secret."""

import argparse
import hashlib
import json
import os
import re
import secrets
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--output", type=Path, help="New mode-0600 credential file (required for issuance)")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--revoke", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.id):
        parser.error("Invalid key id")
    records = json.loads(args.keys.read_text()) if args.keys.exists() else []
    existing = next((record for record in records if record["id"] == args.id), None)
    if args.revoke:
        if not existing:
            parser.error("Unknown key id")
        existing["enabled"] = False
    else:
        if existing or not args.output:
            parser.error("Use a new key id and --output for issuance")
        token = secrets.token_urlsafe(48)
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as file:
            file.write(token + "\n")
        records.append(
            {
                "id": args.id,
                "sha256": hashlib.sha256(token.encode()).hexdigest(),
                "enabled": True,
                "datasets": args.dataset,
            }
        )
    temporary = args.keys.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
    with os.fdopen(fd, "w") as file:
        file.write(json.dumps(records, indent=2))
    if args.keys.exists():
        stat = args.keys.stat()
        os.chown(temporary, stat.st_uid, stat.st_gid)
    temporary.replace(args.keys)
    print("Key revoked" if args.revoke else f"Key written to {args.output}; secret not displayed")


if __name__ == "__main__":
    main()
