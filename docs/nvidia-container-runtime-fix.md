# NVIDIA Container Runtime Fix — `libnvidia-ml.so.1` Not Found

## Problem

When running `docker-compose -f docker-compose-distributed.yml up --build -d`,
containers with GPU access failed to start with:

```
Auto-detected mode as 'legacy'
nvidia-container-cli: initialization error: load library failed:
libnvidia-ml.so.1: cannot open shared object file: no such file or directory
```

The library existed on the host (`/lib/x86_64-linux-gnu/libnvidia-ml.so.1`) and
was registered in ldconfig's cache. The container images were fine. The issue was
entirely in the NVIDIA Container Runtime configuration.

**Environment:**
- Host OS: Ubuntu 22.04
- NVIDIA driver: 580.126.09 (RTX 4070 Ti SUPER)
- Docker: 29.1.3
- nvidia-container-toolkit installed (not the deprecated nvidia-docker2)

---

## Root Cause Chain

### 1. `ldconfig` path was wrong in runtime config

`/etc/nvidia-container-runtime/config.toml` had:

```toml
[nvidia-container-cli]
ldconfig = "@/sbin/ldconfig.real"
```

The `@` prefix means "run this binary from the **host** filesystem". The problem:
`/sbin/ldconfig.real` does **not** exist on this host — only `/sbin/ldconfig` does.
(`ldconfig.real` is a Debian/Ubuntu artifact present only when a wrapper script
replaces the original binary, which was not the case here.)

### 2. docker-compose used the OCI hook approach, which CDI mode does not support

The original compose file used:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["0"]
          capabilities: [gpu]
```

This triggers the `nvidia-container-runtime-hook` (an OCI prestart hook). In CDI
mode this is explicitly disallowed; the runtime returns:

```
invoking the NVIDIA Container Runtime Hook directly is not supported.
Please use the NVIDIA Container Runtime (--runtime=nvidia flag) instead.
```

### 3. `expert_worker` module missing from Docker image

The `Dockerfile` only copied `app/` and `moe_router/`, but the worker service ran:

```
uvicorn expert_worker.main:app ...
```

`expert_worker/` was never added to the image, causing `ModuleNotFoundError`.

---

## Fixes Applied

### Fix 1 — Correct the ldconfig path

Edit `/etc/nvidia-container-runtime/config.toml`:

```bash
sudo sed -i 's|ldconfig = "@/sbin/ldconfig.real"|ldconfig = "@/sbin/ldconfig"|' \
    /etc/nvidia-container-runtime/config.toml
```

Key: keep the `@` prefix (it means "host path"), but point to `/sbin/ldconfig`
instead of the non-existent `/sbin/ldconfig.real`.

### Fix 2 — Switch the NVIDIA runtime to CDI mode

```bash
sudo sed -i 's/^mode = "auto"/mode = "cdi"/' \
    /etc/nvidia-container-runtime/config.toml
```

Generate the CDI spec if not already present:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# Verify: should list nvidia.com/gpu=0, nvidia.com/gpu=all, etc.
sudo nvidia-ctk cdi list
```

Enable CDI spec directory in Docker's daemon:

```bash
sudo tee /etc/docker/daemon.json > /dev/null << 'EOF'
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "cdi-spec-dirs": ["/etc/cdi"]
}
EOF

sudo systemctl restart docker
```

### Fix 3 — Use `runtime: nvidia` in docker-compose (required for CDI mode)

Replace `deploy.resources.reservations.devices` with `runtime: nvidia` on each
GPU-using service. GPU visibility is controlled by the `NVIDIA_VISIBLE_DEVICES`
environment variable already set in the compose file.

```yaml
# Before (OCI hook approach — does not work in CDI mode):
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["0"]
          capabilities: [gpu]

# After (runtime approach — works with CDI mode):
runtime: nvidia
```

### Fix 4 — Add `expert_worker/` to the Dockerfile

```dockerfile
# Before:
COPY app/ ./app/
COPY moe_router/ ./moe_router/

# After:
COPY app/ ./app/
COPY moe_router/ ./moe_router/
COPY expert_worker/ ./expert_worker/
```

---

## Final Config State

**`/etc/nvidia-container-runtime/config.toml` (relevant lines):**

```toml
[nvidia-container-cli]
ldconfig = "@/sbin/ldconfig"   # host-relative, points to real binary

[nvidia-container-runtime]
mode = "cdi"                   # use CDI spec instead of legacy hook
```

**`/etc/docker/daemon.json`:**

```json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "cdi-spec-dirs": ["/etc/cdi"]
}
```

---

## Quick Verification

After applying all fixes, verify GPU is visible inside a container:

```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi
```
