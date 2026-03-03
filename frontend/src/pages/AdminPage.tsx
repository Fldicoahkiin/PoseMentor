import React, { useState } from 'react';
import { Settings, Server, Activity, ArrowRight, Play, LayoutGrid } from 'lucide-react';
import { Button } from '../components/ui/Button';

export default function AdminPage() {
    const [activeTab, setActiveTab] = useState('overview');

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="glass-card rounded-2xl relative overflow-hidden bg-white/60">
                <div className="absolute top-0 inset-x-0 h-1 bg-zinc-900" />
                <div className="p-8 pb-10">
                    <div className="flex justify-between items-start">
                        <div>
                            <h1 className="text-3xl font-extrabold tracking-tight text-zinc-900 flex items-center gap-3 drop-shadow-sm">
                                <LayoutGrid className="text-zinc-500" size={32} /> 控制台大盘
                            </h1>
                            <p className="text-zinc-500 font-medium mt-3 tracking-wide max-w-2xl leading-relaxed">
                                统一调度系统各个作业管道 (Pipeline) — 包含数据集预处理、多机位格式化、模型训练与全链路离线评估打分。
                            </p>
                        </div>

                        <div className="bg-white px-5 py-3 rounded-xl border border-zinc-200 shadow-sm flex items-center gap-3">
                            <div className="relative flex h-3 w-3">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                            </div>
                            <span className="font-bold text-zinc-700 uppercase tracking-widest text-xs">Engine Online</span>
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="bg-zinc-50 border border-zinc-200 rounded-xl p-5 hover:border-zinc-300 transition-colors shadow-sm">
                    <div className="text-zinc-400 uppercase text-xs font-bold tracking-widest mb-2 flex justify-between">
                        待处理 <Server size={14} />
                    </div>
                    <div className="text-3xl font-black text-zinc-800">12</div>
                </div>
                <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 shadow-lg shadow-zinc-900/10">
                    <div className="text-zinc-400 uppercase text-xs font-bold tracking-widest mb-2 flex justify-between">
                        运行中 <Activity size={14} className="text-sky-400" />
                    </div>
                    <div className="text-3xl font-black text-white">3</div>
                </div>
                <div className="bg-zinc-50 border border-zinc-200 rounded-xl p-5 hover:border-zinc-300 transition-colors shadow-sm">
                    <div className="text-zinc-400 uppercase text-xs font-bold tracking-widest mb-2">已成功</div>
                    <div className="text-3xl font-black text-emerald-600">84</div>
                </div>
                <div className="bg-zinc-50 border border-zinc-200 rounded-xl p-5 hover:border-zinc-300 transition-colors shadow-sm">
                    <div className="text-zinc-400 uppercase text-xs font-bold tracking-widest mb-2">失败</div>
                    <div className="text-3xl font-black text-rose-600">1</div>
                </div>
            </div>

            <div className="grid grid-cols-12 gap-6">
                <div className="col-span-12 xl:col-span-3">
                    <div className="glass-card bg-white border-zinc-200 rounded-2xl flex flex-col overflow-hidden shadow-sm">
                        <div className="px-5 py-4 border-b border-zinc-100 font-bold text-zinc-800 bg-zinc-50/50 uppercase tracking-widest text-xs">
                            管道入口
                        </div>

                        <div className="flex flex-col p-2 gap-1">
                            {['overview', 'data', 'pose', 'train', 'eval'].map((tab, idx) => {
                                const names = ['总览信息', '数据下载预处理', '关键点特征提取', '模型训练引擎', '离线全景评估'];
                                const active = activeTab === tab;
                                return (
                                    <button
                                        key={tab}
                                        onClick={() => setActiveTab(tab)}
                                        className={`text-left px-4 py-3 rounded-xl font-semibold text-sm transition-all duration-200 flex justify-between items-center ${active ? 'bg-zinc-900 text-white shadow-md' : 'text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900'
                                            }`}
                                    >
                                        {names[idx]}
                                        {active && <ArrowRight size={16} className="text-zinc-400" />}
                                    </button>
                                )
                            })}
                        </div>
                    </div>
                </div>

                <div className="col-span-12 xl:col-span-9 glass-card bg-white p-8 rounded-2xl border-zinc-200 min-h-[500px] shadow-sm">
                    {activeTab === 'overview' && (
                        <div>
                            <h3 className="text-xl font-bold text-zinc-900 mb-6 flex items-center gap-3">
                                <Activity className="text-zinc-400" />
                                任务执行队列
                            </h3>
                            <div className="border border-zinc-200 rounded-xl overflow-hidden bg-zinc-50">
                                <table className="w-full text-left font-medium">
                                    <thead className="bg-zinc-100/80 text-zinc-500 text-xs uppercase tracking-widest">
                                        <tr>
                                            <th className="px-5 py-4 font-bold">作业 ID</th>
                                            <th className="px-5 py-4 font-bold">模块类型</th>
                                            <th className="px-5 py-4 font-bold">状态</th>
                                            <th className="px-5 py-4 font-bold">耗时</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-zinc-200 text-zinc-700 text-sm bg-white">
                                        <tr className="hover:bg-zinc-50 transition-colors">
                                            <td className="px-5 py-4 font-mono text-zinc-500">task_a79f2b</td>
                                            <td className="px-5 py-4 font-semibold">关键点特征提取</td>
                                            <td className="px-5 py-4"><span className="bg-sky-100 text-sky-700 px-3 py-1 rounded-full text-xs font-bold tracking-wide">RUNNING</span></td>
                                            <td className="px-5 py-4 font-mono">1m 24s</td>
                                        </tr>
                                        <tr className="hover:bg-zinc-50 transition-colors">
                                            <td className="px-5 py-4 font-mono text-zinc-500">task_13be4c</td>
                                            <td className="px-5 py-4 font-semibold">数据下载预处理</td>
                                            <td className="px-5 py-4"><span className="bg-emerald-100 text-emerald-700 px-3 py-1 rounded-full text-xs font-bold tracking-wide">CLEARED</span></td>
                                            <td className="px-5 py-4 font-mono">12m 04s</td>
                                        </tr>
                                        <tr className="hover:bg-zinc-50 transition-colors">
                                            <td className="px-5 py-4 font-mono text-zinc-500">task_6bfe90</td>
                                            <td className="px-5 py-4 font-semibold">模型训练引擎</td>
                                            <td className="px-5 py-4"><span className="bg-rose-100 text-rose-700 px-3 py-1 rounded-full text-xs font-bold tracking-wide">FAILED</span></td>
                                            <td className="px-5 py-4 font-mono">2h 14m</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {activeTab === 'data' && (
                        <div className="max-w-xl">
                            <h3 className="text-xl font-bold text-zinc-900 mb-6">新建：数据预处理任务</h3>
                            <form className="space-y-6">
                                <div>
                                    <label className="block text-xs font-bold text-zinc-500 uppercase tracking-widest mb-2 pl-1">配置文件路径 (data.yaml)</label>
                                    <input type="text" defaultValue="configs/data.yaml" className="w-full bg-zinc-50 border-zinc-200 text-zinc-900 shadow-sm rounded-xl px-4 py-2.5 font-mono text-sm" />
                                </div>
                                <div className="space-y-3 p-5 bg-zinc-50 rounded-xl border border-zinc-200">
                                    <label className="flex items-center gap-3 font-medium text-zinc-700 cursor-pointer">
                                        <input type="checkbox" defaultChecked className="w-5 h-5 rounded text-zinc-900 border-zinc-300 focus:ring-zinc-900 disabled:opacity-50" />
                                        同意下载相关数据集证书许可协议
                                    </label>
                                    <label className="flex items-center gap-3 font-medium text-zinc-700 cursor-pointer">
                                        <input type="checkbox" defaultChecked className="w-5 h-5 rounded text-zinc-900 border-zinc-300 focus:ring-zinc-900 disabled:opacity-50" />
                                        提取核心关键帧注释
                                    </label>
                                </div>
                                <Button className="w-full bg-zinc-900 hover:bg-zinc-800 text-white rounded-xl h-12 flex gap-2">
                                    <Play size={18} fill="currentColor" /> 投递系统执行
                                </Button>
                            </form>
                        </div>
                    )}

                    {activeTab === 'pose' && (
                        <div className="max-w-xl animate-in fade-in zoom-in-95 duration-200">
                            <h3 className="text-xl font-bold text-zinc-900 mb-6">配置：关键点特征提取 (YOLO11)</h3>
                            <form className="space-y-6">
                                <div>
                                    <label className="block text-xs font-bold text-zinc-500 uppercase tracking-widest mb-2 pl-1">预训练模型权重</label>
                                    <select className="w-full bg-zinc-50 border-zinc-200 text-zinc-900 shadow-sm rounded-xl px-4 py-3 font-medium text-sm focus:border-zinc-500 focus:ring-zinc-500 transition-all">
                                        <option value="yolo11l-pose.pt">yolo11l-pose.pt (精确优先 - 推荐)</option>
                                        <option value="yolo11m-pose.pt">yolo11m-pose.pt (平衡)</option>
                                        <option value="yolo11n-pose.pt">yolo11n-pose.pt (速度优先)</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-xs font-bold text-zinc-500 uppercase tracking-widest mb-2 pl-1">推理设备 (Device)</label>
                                    <div className="flex gap-4">
                                        <label className="flex-1 flex items-center justify-center gap-2 p-3 border border-zinc-200 rounded-xl cursor-pointer hover:bg-zinc-50 transition-colors">
                                            <input type="radio" name="device" defaultChecked className="text-zinc-900 focus:ring-zinc-900 w-4 h-4" />
                                            <span className="font-bold text-zinc-700">MPS / GPU</span>
                                        </label>
                                        <label className="flex-1 flex items-center justify-center gap-2 p-3 border border-zinc-200 rounded-xl cursor-pointer hover:bg-zinc-50 transition-colors">
                                            <input type="radio" name="device" className="text-zinc-900 focus:ring-zinc-900 w-4 h-4" />
                                            <span className="font-bold text-zinc-700">CPU</span>
                                        </label>
                                    </div>
                                </div>
                                <Button className="w-full bg-zinc-900 hover:bg-zinc-800 text-white rounded-xl h-12 flex gap-2">
                                    <Play size={18} fill="currentColor" /> 启动特征提取管线
                                </Button>
                            </form>
                        </div>
                    )}

                    {activeTab === 'train' && (
                        <div className="max-w-2xl animate-in fade-in zoom-in-95 duration-200">
                            <h3 className="text-xl font-bold text-zinc-900 mb-6">配置：3D Lift网络训练</h3>
                            <form className="space-y-6">
                                <div className="grid grid-cols-2 gap-6">
                                    <div>
                                        <label className="block text-xs font-bold text-zinc-500 uppercase tracking-widest mb-2 pl-1">训练轮次 (Epochs)</label>
                                        <input type="number" defaultValue="50" className="w-full bg-zinc-50 border-zinc-200 text-zinc-900 shadow-sm rounded-xl px-4 py-2.5 font-mono text-sm" />
                                    </div>
                                    <div>
                                        <label className="block text-xs font-bold text-zinc-500 uppercase tracking-widest mb-2 pl-1">批次大小 (Batch Size)</label>
                                        <input type="number" defaultValue="128" className="w-full bg-zinc-50 border-zinc-200 text-zinc-900 shadow-sm rounded-xl px-4 py-2.5 font-mono text-sm" />
                                    </div>
                                    <div>
                                        <label className="block text-xs font-bold text-zinc-500 uppercase tracking-widest mb-2 pl-1">学习率 (Learning Rate)</label>
                                        <input type="text" defaultValue="1e-3" className="w-full bg-zinc-50 border-zinc-200 text-zinc-900 shadow-sm rounded-xl px-4 py-2.5 font-mono text-sm" />
                                    </div>
                                    <div>
                                        <label className="block text-xs font-bold text-zinc-500 uppercase tracking-widest mb-2 pl-1">输入通道维度</label>
                                        <input type="number" defaultValue="34" disabled className="w-full bg-zinc-100 border-zinc-200 text-zinc-500 shadow-sm rounded-xl px-4 py-2.5 font-mono text-sm cursor-not-allowed" />
                                    </div>
                                </div>

                                <div className="p-5 bg-zinc-50 rounded-xl border border-zinc-200 flex items-start gap-3">
                                    <Settings className="text-zinc-400 shrink-0 mt-0.5" size={20} />
                                    <div>
                                        <h4 className="font-bold text-zinc-800 text-sm">增量训练模式</h4>
                                        <p className="text-zinc-500 text-xs mt-1 leading-relaxed">检测到 outputs/checkpoints 下已有权重，若勾选下方选项，则加载最新检查点继续训练以延长 Epoch，而非从头初始化网络参数。</p>
                                        <label className="flex items-center gap-3 font-medium text-zinc-700 cursor-pointer mt-3">
                                            <input type="checkbox" defaultChecked className="w-4 h-4 rounded text-zinc-900 border-zinc-300 focus:ring-zinc-900" />
                                            从中断点恢复训练 (Resume Training)
                                        </label>
                                    </div>
                                </div>

                                <Button className="w-full bg-zinc-900 hover:bg-zinc-800 text-white rounded-xl h-12 flex gap-2">
                                    <Play size={18} fill="currentColor" /> 下发分布式训练任务
                                </Button>
                            </form>
                        </div>
                    )}

                    {activeTab === 'eval' && (
                        <div className="max-w-xl animate-in fade-in zoom-in-95 duration-200">
                            <h3 className="text-xl font-bold text-zinc-900 mb-6">全景视图与评估报告</h3>

                            <div className="flex flex-col gap-4">
                                <div className="border border-zinc-200 rounded-xl p-5 flex justify-between items-center hover:shadow-md transition-shadow bg-white">
                                    <div className="flex items-center gap-4">
                                        <div className="w-10 h-10 rounded-full bg-zinc-100 flex items-center justify-center text-zinc-500">
                                            <Activity size={20} />
                                        </div>
                                        <div>
                                            <div className="font-bold text-zinc-900">执行 DTW 时序对齐精度测试</div>
                                            <div className="text-xs text-zinc-500 mt-1">遍历测试集输出评价指标矩阵 (MPJPE / PA-MPJPE)</div>
                                        </div>
                                    </div>
                                    <Button variant="outline" className="border-zinc-300 text-zinc-700 font-bold hover:bg-zinc-100">运行</Button>
                                </div>

                                <div className="border border-zinc-200 rounded-xl p-5 flex justify-between items-center hover:shadow-md transition-shadow bg-white">
                                    <div className="flex items-center gap-4">
                                        <div className="w-10 h-10 rounded-full bg-zinc-100 flex items-center justify-center text-zinc-500">
                                            <ArrowRight size={20} />
                                        </div>
                                        <div>
                                            <div className="font-bold text-zinc-900">导出生产环境 ONNX 模型</div>
                                            <div className="text-xs text-zinc-500 mt-1">将当前 PyTorch Checkpoint 量化导出给前端推理使用</div>
                                        </div>
                                    </div>
                                    <Button variant="outline" className="border-zinc-300 text-zinc-700 font-bold hover:bg-zinc-100">导出</Button>
                                </div>

                                <div className="mt-4 p-4 rounded-xl bg-orange-50 border border-orange-200 text-orange-800 text-sm">
                                    <strong>提示：</strong> 目前在系统中没有找到有效的缓存模型快照，这可能导致在线评分功能短时间内表现欠佳。建议优先执行一次训练全生命周期。
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
