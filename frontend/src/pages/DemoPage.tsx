import React, { useState, useRef, useEffect } from 'react';
import { Camera, Upload, Video, RefreshCw, AlertCircle, Volume2, ChartBar } from 'lucide-react';
import { Button } from '../components/ui/Button';

export default function DemoPage() {
    const [activeTab, setActiveTab] = useState<'webcam' | 'video'>('webcam');
    const [style, setStyle] = useState('gBR');
    const [initError, setInitError] = useState('');

    // Fake state for webcam demo
    const webcamRef = useRef<HTMLVideoElement>(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const [score, setScore] = useState(0);

    useEffect(() => {
        // Simulated mock init
        setInitError('');
    }, []);

    const handleStartCam = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (webcamRef.current) {
                webcamRef.current.srcObject = stream;
                setIsStreaming(true);
            }
        } catch (err) {
            console.error(err);
            setInitError('无法访问摄像头，请检查权限。');
        }
    };

    const stopCam = () => {
        if (webcamRef.current && webcamRef.current.srcObject) {
            const tracks = (webcamRef.current.srcObject as MediaStream).getTracks();
            tracks.forEach(t => t.stop());
            webcamRef.current.srcObject = null;
        }
        setIsStreaming(false);
    }

    // Generate fake score
    useEffect(() => {
        if (!isStreaming) return;
        const interval = setInterval(() => {
            setScore(75 + Math.random() * 20); // 75~95
        }, 1000);
        return () => clearInterval(interval);
    }, [isStreaming]);

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Header Panel */}
            <div className="glass-card rounded-2xl relative overflow-hidden bg-white/60">
                <div className="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-zinc-800 to-zinc-400" />
                <div className="p-8 pb-10">
                    <div className="flex items-center gap-4 mb-4">
                        <div className="bg-zinc-100 p-3 rounded-full text-zinc-900 border border-zinc-200">
                            <Camera size={28} />
                        </div>
                        <div>
                            <h1 className="text-3xl font-extrabold tracking-tight text-zinc-900 drop-shadow-sm">
                                单摄像头动作教学平台
                            </h1>
                            <p className="text-zinc-500 font-medium mt-1 tracking-wide">
                                实时采集视觉关键点 · 映射3D骨骼 · AI分析打分及纠错
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {initError && (
                <div className="p-4 bg-orange-50 border border-orange-200 rounded-xl text-orange-700 font-medium flex gap-3 items-center">
                    <AlertCircle size={20} />
                    <span>{initError}</span>
                </div>
            )}

            {/* Main Tabs */}
            <div className="flex items-center gap-2 mb-2 p-1 bg-zinc-200/50 rounded-xl w-fit">
                <button
                    onClick={() => setActiveTab('webcam')}
                    className={`px-5 py-2.5 rounded-lg font-semibold text-sm transition-all duration-200 ${activeTab === 'webcam'
                            ? 'bg-white text-zinc-900 shadow-sm'
                            : 'text-zinc-500 hover:text-zinc-700 hover:bg-zinc-200/50'
                        }`}
                >
                    <div className="flex items-center gap-2">
                        <Video size={16} /> 直播流评估
                    </div>
                </button>
                <button
                    onClick={() => setActiveTab('video')}
                    className={`px-5 py-2.5 rounded-lg font-semibold text-sm transition-all duration-200 ${activeTab === 'video'
                            ? 'bg-white text-zinc-900 shadow-sm'
                            : 'text-zinc-500 hover:text-zinc-700 hover:bg-zinc-200/50'
                        }`}
                >
                    <div className="flex items-center gap-2">
                        <Upload size={16} /> 视频上传分析
                    </div>
                </button>
            </div>

            {/* Webcam Tab */}
            {activeTab === 'webcam' && (
                <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                    <div className="xl:col-span-2 space-y-4">
                        <div className="glass-card bg-white p-6 rounded-2xl border-zinc-200">
                            <div className="flex justify-between items-end mb-4">
                                <div>
                                    <label className="block text-xs font-bold text-zinc-500 uppercase tracking-widest mb-1.5 pl-1">参考舞种</label>
                                    <select
                                        value={style}
                                        onChange={e => setStyle(e.target.value)}
                                        className="pl-3 pr-10 py-2 bg-zinc-50 border-zinc-200 text-zinc-900 shadow-sm focus:border-zinc-500 focus:ring-zinc-500"
                                    >
                                        <option value="gBR">gBR - Breakdance</option>
                                        <option value="gPO">gPO - Popping</option>
                                        <option value="gLO">gLO - Locking</option>
                                    </select>
                                </div>
                                <div className="flex gap-2">
                                    {!isStreaming ? (
                                        <Button onClick={handleStartCam} variant="default" className="bg-zinc-900 hover:bg-zinc-800">
                                            开启摄像头
                                        </Button>
                                    ) : (
                                        <Button onClick={stopCam} variant="destructive">
                                            关闭摄像头
                                        </Button>
                                    )}
                                    <Button variant="outline"><RefreshCw size={16} className="mr-2" /> 重置记录</Button>
                                </div>
                            </div>

                            <div className="aspect-video bg-zinc-900 rounded-xl overflow-hidden relative border border-zinc-200 shadow-inner flex items-center justify-center">
                                <video
                                    ref={webcamRef}
                                    autoPlay
                                    playsInline
                                    muted
                                    className={`w-full h-full object-cover ${!isStreaming ? 'hidden' : ''}`}
                                />
                                {!isStreaming && (
                                    <div className="text-zinc-600 flex flex-col items-center">
                                        <Camera size={48} className="mb-4 opacity-30" />
                                        <p className="font-medium tracking-wide">摄像头未连接，请点击上方开启</p>
                                    </div>
                                )}

                                {/* Simulated overlays */}
                                {isStreaming && (
                                    <div className="absolute top-4 left-4 bg-red-500 text-white text-xs font-bold px-2 py-1 rounded-md flex items-center gap-1.5 shadow-lg">
                                        <div className="w-2 h-2 rounded-full bg-white animate-pulse" /> LIVE
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="space-y-6">
                        <div className="glass-card bg-white p-6 rounded-2xl border-zinc-200 relative overflow-hidden">
                            <div className="absolute top-0 right-0 w-32 h-32 bg-zinc-100 rounded-bl-full -z-10" />
                            <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-2 mb-6">
                                <ChartBar size={16} /> 实时评分面板
                            </h3>

                            <div className="text-center py-6">
                                <div className="text-6xl font-black text-zinc-900 tracking-tighter tabular-nums drop-shadow-sm">
                                    {score > 0 ? score.toFixed(1) : '--'}
                                </div>
                                <div className="text-zinc-500 font-medium mt-2">总体表现分 (Score)</div>
                            </div>

                            <div className="space-y-4 mt-4">
                                <div className="bg-zinc-50 border border-zinc-100 rounded-xl p-4 flex justify-between items-center shadow-sm">
                                    <span className="text-zinc-500 font-medium text-sm">平均 MPJPE</span>
                                    <span className="font-bold text-lg text-zinc-800 tabular-nums">12.4 <span className="text-xs text-zinc-400">mm</span></span>
                                </div>
                                <div className="bg-zinc-50 border border-zinc-100 rounded-xl p-4 flex justify-between items-center shadow-sm">
                                    <span className="text-zinc-500 font-medium text-sm">骨骼角度误差</span>
                                    <span className="font-bold text-lg text-zinc-800 tabular-nums">3.8 <span className="text-xs text-zinc-400">deg</span></span>
                                </div>
                            </div>
                        </div>

                        <div className="glass-card bg-zinc-900 border-zinc-800 text-white p-6 rounded-2xl shadow-xl shadow-zinc-900/20">
                            <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-2 mb-4">
                                <Volume2 size={16} /> 语音提示与纠错
                            </h3>
                            <p className="text-zinc-200 font-medium leading-relaxed mt-2 p-1 bg-zinc-800/50 rounded-lg whitespace-pre-line border border-zinc-700/50">
                                {isStreaming ? (
                                    "注意你的左臂姿态，稍微放低一点，\n保持右腿膝盖弯曲。"
                                ) : (
                                    "等待采集..."
                                )}
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Video Upload Tab (mock) */}
            {activeTab === 'video' && (
                <div className="glass-card bg-white p-8 rounded-2xl border-zinc-200 text-center py-20 flex flex-col items-center">
                    <div className="w-20 h-20 bg-zinc-100 text-zinc-300 rounded-full flex items-center justify-center mb-6 border-2 border-dashed border-zinc-300">
                        <Upload size={32} />
                    </div>
                    <h3 className="text-xl font-bold text-zinc-800 mb-2">拖拽或点击上传视频</h3>
                    <p className="text-zinc-500 font-medium mb-6">支持 MP4, MOV 格式 (最大文件 50MB)</p>
                    <Button className="bg-zinc-900 hover:bg-zinc-800 rounded-full px-8">选择视频文件</Button>
                </div>
            )}

        </div>
    );
}
