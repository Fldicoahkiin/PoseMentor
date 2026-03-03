# PoseMentor Frontend

React + TypeScript + Vite 前端工程。

职责边界：
- 只负责交互与可视化
- 通过后端 API 获取任务和结果
- 不直接调用训练脚本或读写模型文件

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

建议在前端中统一以 `dataset_id` 与后端交互，便于后续接入非 AIST++ 数据源。

## 目录

- `src/pages/`：页面入口
- `src/components/`：通用组件
- `src/lib/`：工具函数
