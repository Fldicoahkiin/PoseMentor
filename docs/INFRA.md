# 基础设施说明

## 组件

- `backend_api.py`：FastAPI 后台接口，负责启动数据、训练、评测、多机位任务。  
- `admin_console.py`：Gradio 管理界面，调用 Backend API 管理任务。  
- `scripts/launch_macos.sh` / `scripts/launch_windows.ps1`：跨平台一键部署脚本。  
- `outputs/job_center/`：任务元数据与日志目录。

## 任务状态存储

- 状态文件：`outputs/job_center/jobs.json`  
- 任务日志：`outputs/job_center/logs/<job_id>.log`

## 任务执行机制

1. 前端提交任务参数到 Backend API。  
2. Backend API 生成命令并写入 JobStore。  
3. JobRunner 使用后台线程执行命令，实时落盘日志。  
4. 前端轮询任务列表与日志。

## 容器部署

`docker/docker-compose.yml` 包含三类服务：
- `backend-api`（任务编排）
- `admin-console`（管理面板）
- `posementor-app`（在线系统）

## 建议

- 线上部署时将 `outputs/job_center` 挂载到持久化卷。  
- 生产环境建议在反向代理层增加鉴权。  
- 如需多用户并发任务，建议引入队列系统（Redis + worker）并保留当前接口层。
