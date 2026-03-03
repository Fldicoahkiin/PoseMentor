import os
import subprocess
import typer

app = typer.Typer(help="PoseMentor CLI - 单摄像头 AI 舞蹈矫正系统管理工具")

@app.command()
def start():
    """
    一键启动整个开发与运行系统（自动启动 Backend API 和 Vite 前端）
    """
    typer.echo("🚀 正在启动 PoseMentor 系统服务...")
    
    # 获取项目根目录，Procfile 放被在项目根目录中
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    procfile_path = os.path.join(project_root, "Procfile")
    
    if not os.path.exists(procfile_path):
        typer.secho(f"❌ 找不到 Procfile 配置: {procfile_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
        
    try:
        # 使用 Honcho 运行 Procfile 中定义的服务进程
        subprocess.run(["honcho", "start", "-f", procfile_path], cwd=project_root)
    except KeyboardInterrupt:
        typer.echo("\n🛑 PoseMentor 服务已停止")
    except Exception as e:
        typer.secho(f"启动失败: {e}", fg=typer.colors.RED)

@app.command()
def version():
    """
    查看 PoseMentor 系统版本
    """
    typer.echo("PoseMentor Version: 0.1.0")
    
def main():
    app()

if __name__ == "__main__":
    main()
