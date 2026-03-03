# PoseMentor Frontend

React + TypeScript + Vite 前端工程。

## 启动

```bash
cd /Users/mac/WorkSpace/Python_Project/posementor/frontend
pnpm install
pnpm dev --host 127.0.0.1 --port 7860
```

## 常用命令

```bash
pnpm lint
pnpm build
pnpm preview
```

## 对接后端

后端默认地址：`http://127.0.0.1:8787`

建议联调时先启动后端：

```bash
cd /Users/mac/WorkSpace/Python_Project/posementor
uv run python backend_api.py
```

## 目录

- `src/pages/`：页面入口
- `src/components/`：通用组件
- `src/lib/`：工具函数
