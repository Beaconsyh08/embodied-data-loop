"""Authenticated evaluation exports and reference-bound oracle prediction imports."""

import getpass
import json
import os
import uuid
from pathlib import Path

from data_platform_cli.client import ClientError


def manage(client, args):
    try:
        location = str(uuid.UUID(args.location or ""))
    except ValueError:
        raise ClientError("evaluation requires --location UUID") from None
    base = f"/api/control/locations/{location}/evaluation"
    link = client.origin + f"/remote/{location}/temporal-caption/evaluation"
    if args.web:
        print(link)
        return 0
    target = Path(args.export_file).expanduser() if args.export_file else None
    if target and (target.exists() or target.is_symlink()):
        raise ClientError("Choose a new export file; existing files are not overwritten")
    oracle = None
    if args.oracle_file:
        if args.episode is None:
            raise ClientError("--oracle-file requires --episode")
        source = Path(args.oracle_file).expanduser()
        if source.stat().st_size > 1024 * 1024:
            raise ClientError("Oracle prediction file exceeds 1 MiB")
        oracle = json.loads(source.read_text())
    client.login(args.username or input("Platform username: "), getpass.getpass("Platform password: "))
    try:
        if oracle is not None:
            record = client.request("GET", base + f"/episodes/{args.episode}")
            client.request(
                "POST",
                base + f"/episodes/{args.episode}",
                {"action": "oracle-import", "revision": record["revision"], "oracle": oracle},
            )
            print("Oracle predictions imported; human semantic assessment is pending.")
        if target:
            value = client.request("GET", base + "/export")
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2)
            print("Evaluation export saved: " + str(target.absolute()))
        else:
            print(json.dumps(client.request("GET", base + "/metrics"), ensure_ascii=True, indent=2))
        print(link)
    finally:
        client.request("POST", "/api/auth/logout", {})
    return 0
