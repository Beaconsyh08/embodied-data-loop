#!/usr/bin/env bash
# Independent API service only. Never restarts or replaces Data Platform services.
set -euo pipefail
SOURCE=${1:?absolute unpacked service release directory}
ADAPTER=${2:?absolute immutable installed Data Platform Agent release directory}
RUNTIME=${3:?existing FLORA runtime JSON}
PUBLIC_IP=${4:?public IPv4 address}
STATE=${5:?dedicated API state directory on the data disk}
test "$(id -u)" = 0
test -x "$ADAPTER/.venv/bin/python"
test -f "$SOURCE/services/flora_api/app.py"
id flora-api >/dev/null 2>&1 || useradd --system --home-dir "$STATE/home" --shell /usr/sbin/nologin flora-api
install -d -o root -g flora-api -m 0750 /etc/flora-api /etc/flora-api/tls
install -d -o flora-api -g flora-api -m 0700 "$STATE" "$STATE/home" "$STATE/tmp"
"$ADAPTER/uv" venv --python "$ADAPTER/.venv/bin/python" "$SOURCE/.venv"
"$ADAPTER/uv" pip install --python "$SOURCE/.venv/bin/python" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple gunicorn==23.0.0
"$SOURCE/.venv/bin/python" - "$ADAPTER" "$RUNTIME" "$STATE" "$SOURCE" <<'PY'
import json,site,subprocess,sys,shutil
from pathlib import Path
adapter,runtime,state,source=sys.argv[1:]
installed=next((Path(adapter)/'.venv/lib').glob('python*/site-packages'))
snapshot=Path('/opt/flora-api/dependencies')/Path(adapter).name/'site-packages'
if not snapshot.exists():
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    temporary=snapshot.with_name('site-packages.incoming')
    subprocess.run(['cp','-al',str(installed),str(temporary)],check=True)
    temporary.rename(snapshot)
(Path(site.getsitepackages()[0])/'platform-adapter.pth').write_text(str(snapshot)+'\n')
config=json.loads(Path(runtime).read_text())
config['gpus']=list(reversed(config.get('gpus') or [config.get('gpu',0)]))
source=Path(source)
caption=source/'caption-runtime'
shutil.copytree(config['caption_code'],caption,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copyfile(source/'services/flora_api/manifest_cache.py',caption/'flora_manifest_cache.py')
models=caption/'batch_utils/models.py'
with models.open('a') as file:
    file.write('\n# Standalone API: retain upstream integrity checks across worker processes.\n'
               'from flora_manifest_cache import enable as _enable_api_verification_cache\n'
               'import os as _api_os\n'
               '_enable_api_verification_cache(globals(), _api_os.environ.get("FLORA_MODEL_VERIFY_CACHE"))\n')
config['caption_code']=str(caption)
Path('/etc/flora-api/runtime.json').write_text(json.dumps(config,indent=2))
access_record=Path('/etc/flora-api/model-access.json')
known=set(json.loads(access_record.read_text())) if access_record.exists() else set()
for name in ('caption_python','reward_python','caption_code','weights'):
    path=Path(config[name])
    target=path.parent.parent if name.endswith('_python') else path
    if str(target) not in known:
        subprocess.run(['setfacl','-Rm','u:flora-api:rX',str(target)],check=True)
        known.add(str(target))
    for parent in path.resolve().parents:
        if parent != Path('/'):
            subprocess.run(['setfacl','-m','u:flora-api:x',str(parent)],check=True)
access_record.write_text(json.dumps(sorted(known)))
access_record.chmod(0o600)
lock=Path(config['lock_dir'])
for parent in [lock,*lock.parents]:
    if parent != Path('/'):
        subprocess.run(['setfacl','-m','u:flora-api:x',str(parent)],check=True)
for gpu in config['gpus']:
    target=lock/f'gpu-{gpu}.lock'
    if not target.is_file(): raise RuntimeError('Shared platform GPU lock must already exist')
    subprocess.run(['setfacl','-m','u:flora-api:rw',str(target)],check=True)
cache=lock/'hf-cache'
cache.mkdir(exist_ok=True)
subprocess.run(['setfacl','-Rm','u:flora-api:rwX',str(cache)],check=True)
subprocess.run(['setfacl','-m','d:u:flora-api:rwX',str(cache)],check=True)
PY
setfacl -m u:flora-api:rx "$SOURCE"
chown root:flora-api /etc/flora-api/runtime.json
chmod 0640 /etc/flora-api/runtime.json
if [ ! -f /etc/flora-api/keys.json ]; then
  install -o root -g flora-api -m 0640 /dev/null /etc/flora-api/keys.json
  echo '[]' > /etc/flora-api/keys.json
fi
if [ ! -f /etc/flora-api/datasets.json ]; then
  install -o root -g flora-api -m 0640 /dev/null /etc/flora-api/datasets.json
  echo '{}' > /etc/flora-api/datasets.json
fi
if [ ! -f /etc/flora-api/tls/server.pem ]; then
  openssl req -x509 -newkey rsa:3072 -nodes -days 3650 \
    -keyout /etc/flora-api/tls/ca.key -out /etc/flora-api/tls/ca.pem \
    -subj '/CN=FLORA API Private CA' -addext 'basicConstraints=critical,CA:TRUE' 2>/dev/null
  openssl req -new -newkey rsa:2048 -nodes \
    -keyout /etc/flora-api/tls/server.key -out /etc/flora-api/tls/server.csr \
    -subj "/CN=$PUBLIC_IP" 2>/dev/null
  printf 'subjectAltName=IP:%s\nbasicConstraints=critical,CA:FALSE\nextendedKeyUsage=serverAuth\n' \
    "$PUBLIC_IP" > /etc/flora-api/tls/extensions.cnf
  openssl x509 -req -days 365 -in /etc/flora-api/tls/server.csr \
    -CA /etc/flora-api/tls/ca.pem -CAkey /etc/flora-api/tls/ca.key -CAcreateserial \
    -extfile /etc/flora-api/tls/extensions.cnf -out /etc/flora-api/tls/server.pem 2>/dev/null
fi
chmod 0600 /etc/flora-api/tls/ca.key
chown root:flora-api /etc/flora-api/tls/server.key /etc/flora-api/tls/server.pem
chmod 0640 /etc/flora-api/tls/server.key /etc/flora-api/tls/server.pem
cat > /etc/flora-api/service.env <<EOF
FLORA_API_STATE=$STATE
FLORA_API_KEYS=/etc/flora-api/keys.json
FLORA_API_DATASETS=/etc/flora-api/datasets.json
DATA_PLATFORM_FLORA_RUNTIME=/etc/flora-api/runtime.json
FLORA_API_CONCURRENCY=8
FLORA_MODEL_VERIFY_CACHE=$STATE/model-manifest-cache
DATA_PLATFORM_VIDEO_THREADS=2
HOME=$STATE/home
TMPDIR=$STATE/tmp
OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=4
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
PYTHONDONTWRITEBYTECODE=1
VLLM_HOST_IP=127.0.0.1
GLOO_SOCKET_IFNAME=lo
NCCL_SOCKET_IFNAME=lo
EOF
for component in web worker; do
  if [ "$component" = web ]; then
    COMMAND="$SOURCE/.venv/bin/gunicorn --workers 2 --worker-class gthread --threads 8 --bind 0.0.0.0:8448 --timeout 3600 --limit-request-line 2048 --certfile /etc/flora-api/tls/server.pem --keyfile /etc/flora-api/tls/server.key services.flora_api.app:create_app()"
    EXTRA=''
    MEMORY_LIMIT=2G
    CPU_LIMIT=400%
  else
    COMMAND="$SOURCE/.venv/bin/python -m services.flora_api.worker"
    EXTRA='IPAddressDeny=any
IPAddressAllow=localhost'
    MEMORY_LIMIT=700G
    CPU_LIMIT=3200%
  fi
  cat > "/etc/systemd/system/flora-api-$component.service" <<EOF
[Unit]
Description=Standalone FLORA API $component
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=flora-api
Group=flora-api
WorkingDirectory=$SOURCE
EnvironmentFile=/etc/flora-api/service.env
ExecStart=$COMMAND
Restart=on-failure
RestartSec=5
KillMode=control-group
TimeoutStopSec=60
UMask=0077
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=$STATE
ReadWritePaths=$("$SOURCE/.venv/bin/python" -c 'import json; print(json.load(open("/etc/flora-api/runtime.json"))["lock_dir"])')
PrivateTmp=true
Nice=5
IOSchedulingClass=best-effort
IOSchedulingPriority=6
TasksMax=8192
MemoryMax=$MEMORY_LIMIT
CPUQuota=$CPU_LIMIT
LimitNOFILE=65536
$EXTRA

[Install]
WantedBy=multi-user.target
EOF
done
systemctl daemon-reload
echo 'Prepared. Issue keys, authorize datasets, then enable flora-api-worker and flora-api-web.'
