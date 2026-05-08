# CLAUDE.md

详细的项目约定和架构文档见 [AGENTS.md](./AGENTS.md)。

## Quick Reference

- Python 3.11+, 包管理用 uv
- `./pm up` 启动全栈（后端 + 前端）
- `./pm down` 停止服务
- `./pm doctor` 检查环境依赖
- `uv run pytest tests/` 运行测试
- `uv run ruff check src/ backend_api.py` 执行 lint
- `cd frontend && pnpm build` 前端构建
- UI 文本使用中文 (zh-CN)
- 路由统一注册在 `APIRouter` 上，`/` 和 `/api` 双前缀挂载
- Job status 枚举: `queued`, `running`, `succeeded`, `failed`
