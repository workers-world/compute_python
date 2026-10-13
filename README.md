# compute_python

策略 / 数值计算引擎（FastAPI + Optuna / NumPy / Joblib）。

**本仓只产 Docker 镜像**，不挂 Cloudflare Worker。对外访问入口是 **[cpt1](../cpt1/)**（Bearer + CF Containers 透传）。

## 角色对照

| 层 | 仓 | 职责 |
|----|-----|------|
| 镜像 | `compute_python`（本仓） | Python 实现 → `ghcr.io/workers-world/compute_python` |
| 访问 | `cpt1` | 鉴权、DO 路由、Container 透传 |
| 业务 | `strategy-monitor` 等 | 经 `SVC_CPT1` 调用，不直连镜像 |

对齐 fund 侧：`install_fund_name` ↔ 本仓，`fund-info-worker` ↔ `cpt1`。

## API（容器内）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/optimize` | Optuna / TPE / GRID 寻优 |
| POST | `/api/simulate` | 单组参数回测 |
| POST | `/api/scenario-table` | 场景网格 |
| POST | `/api/advice` | 即时交易建议 |

`cpt1` 对外映射为 `/cpt1/v1/*`（见该仓 README）。

## 镜像发布

GitHub Actions [`.github/workflows/build-image.yml`](.github/workflows/build-image.yml)：

- 触发：`push` 到 `dev_00_02_00` / `master`、`workflow_dispatch`、周日 cron
- 镜像：`ghcr.io/workers-world/compute_python:latest`

```bash
# 本地构建
docker build -t compute_python:local .
docker run --rm -p 8000:8000 compute_python:local
curl -s http://127.0.0.1:8000/api/health
```

本机开发也可用 `./start.sh`（默认 `:8002`，避让容器 `:8000`）。

## 发布节奏

1. 改本仓 Python → push → GHA 推 GHCR
2. 在 **cpt1** 执行全量 `npm run deploy`（重建 Container）
3. 业务 Worker（如 strategy-monitor）通常只需 `--containers-rollout=none` 或不动

## 依赖

见 [`requirements.txt`](requirements.txt)（无 pandas / matplotlib；仅 stdout 日志）。
