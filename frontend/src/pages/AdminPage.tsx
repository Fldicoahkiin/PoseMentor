import { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, Database, FileText, Play, RefreshCw, Settings, Wrench } from 'lucide-react';
import { Button } from '../components/ui/Button';
import {
  createDataPrepareJob,
  createEvaluateJob,
  createMultiviewJob,
  createPoseExtractJob,
  createTrainJob,
  fetchDatasets,
  fetchJobLog,
  fetchJobs,
  type DatasetItem,
  type JobItem,
} from '../lib/api';

type TabKey = 'overview' | 'data' | 'extract' | 'train' | 'multiview' | 'evaluate';

export default function AdminPage() {
  const [tab, setTab] = useState<TabKey>('overview');
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [jobs, setJobs] = useState<JobItem[]>([]);
  const [selectedJobId, setSelectedJobId] = useState('');
  const [jobLog, setJobLog] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const [datasetId, setDatasetId] = useState('aistpp');

  const [dataConfig, setDataConfig] = useState('configs/data.yaml');
  const [downloadVideos, setDownloadVideos] = useState(false);
  const [videoLimit, setVideoLimit] = useState(80);
  const [agreeLicense, setAgreeLicense] = useState(false);

  const [extractConfig, setExtractConfig] = useState('configs/data.yaml');
  const [weights, setWeights] = useState('yolo11m-pose.pt');
  const [extractConf, setExtractConf] = useState(0.35);
  const [maxVideos, setMaxVideos] = useState(0);
  const [inputDir, setInputDir] = useState('');
  const [outDir, setOutDir] = useState('');
  const [recursive, setRecursive] = useState(false);

  const [trainConfig, setTrainConfig] = useState('configs/train.yaml');
  const [yolo2dDir, setYolo2dDir] = useState('');
  const [gt3dDir, setGt3dDir] = useState('');
  const [artifactDir, setArtifactDir] = useState('');
  const [exportOnnx, setExportOnnx] = useState(true);

  const [multiviewConfig, setMultiviewConfig] = useState('configs/multiview.yaml');
  const [limitSessions, setLimitSessions] = useState(20);

  const [evalInputDir, setEvalInputDir] = useState('data/raw/aistpp/videos');
  const [evalStyle, setEvalStyle] = useState('gBR');
  const [evalMaxVideos, setEvalMaxVideos] = useState(20);
  const [evalOutput, setEvalOutput] = useState('outputs/eval/summary.csv');

  const refresh = useCallback(async () => {
    try {
      const [datasetsResp, jobsResp] = await Promise.all([fetchDatasets(), fetchJobs()]);
      setDatasets(datasetsResp);
      setJobs(jobsResp);
    } catch (err) {
      console.error(err);
      setError('无法连接后端，请先启动 backend_api.py');
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => {
      void refresh();
    }, 6000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (!selectedJobId) {
      setJobLog('');
      return;
    }
    const run = async () => {
      try {
        const logText = await fetchJobLog(selectedJobId);
        setJobLog(logText || '(暂无日志输出)');
      } catch {
        setJobLog('日志读取失败');
      }
    };
    void run();
  }, [selectedJobId]);

  const jobStats = useMemo(() => {
    const queued = jobs.filter((job) => job.status === 'queued').length;
    const running = jobs.filter((job) => job.status === 'running').length;
    const succeeded = jobs.filter((job) => job.status === 'succeeded').length;
    const failed = jobs.filter((job) => job.status === 'failed').length;
    return { queued, running, succeeded, failed };
  }, [jobs]);

  const runTask = useCallback(async (runner: () => Promise<string>) => {
    setBusy(true);
    setMessage('');
    setError('');
    try {
      const jobId = await runner();
      setMessage(`任务已提交：${jobId}`);
      setSelectedJobId(jobId);
      await refresh();
    } catch (err: unknown) {
      console.error(err);
      const detail =
        typeof err === 'object' &&
        err !== null &&
        'response' in err &&
        typeof (err as { response?: { data?: { detail?: unknown } } }).response?.data?.detail === 'string'
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : undefined;
      setError(detail || '任务提交失败，请检查参数');
    } finally {
      setBusy(false);
    }
  }, [refresh]);

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <section className="rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-black tracking-tight text-zinc-900">任务控制台</h1>
            <p className="mt-2 text-zinc-600">数据准备、特征提取、训练与评测任务统一在这里下发和跟踪。</p>
          </div>
          <Button variant="outline" onClick={() => void refresh()} className="gap-2">
            <RefreshCw size={16} /> 刷新任务
          </Button>
        </div>
      </section>

      {(message || error) && (
        <div className={`rounded-xl border px-4 py-3 text-sm font-medium ${error ? 'border-rose-200 bg-rose-50 text-rose-700' : 'border-emerald-200 bg-emerald-50 text-emerald-700'}`}>
          {error || message}
        </div>
      )}

      <section className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <div className="rounded-xl border border-zinc-200 bg-white p-4"><p className="text-xs text-zinc-500">待处理</p><p className="mt-2 text-2xl font-black text-zinc-900">{jobStats.queued}</p></div>
        <div className="rounded-xl border border-zinc-200 bg-white p-4"><p className="text-xs text-zinc-500">运行中</p><p className="mt-2 text-2xl font-black text-zinc-900">{jobStats.running}</p></div>
        <div className="rounded-xl border border-zinc-200 bg-white p-4"><p className="text-xs text-zinc-500">成功</p><p className="mt-2 text-2xl font-black text-zinc-900">{jobStats.succeeded}</p></div>
        <div className="rounded-xl border border-zinc-200 bg-white p-4"><p className="text-xs text-zinc-500">失败</p><p className="mt-2 text-2xl font-black text-zinc-900">{jobStats.failed}</p></div>
      </section>

      <section className="grid grid-cols-1 gap-6 xl:grid-cols-4">
        <div className="rounded-2xl border border-zinc-200 bg-white p-3 shadow-sm">
          {([
            ['overview', '任务总览', Activity],
            ['data', '数据准备', Database],
            ['extract', '关键点提取', Wrench],
            ['train', '模型训练', Settings],
            ['multiview', '多机位处理', Wrench],
            ['evaluate', '评测导出', FileText],
          ] as [TabKey, string, React.ComponentType<{ size?: number }>][])
            .map(([key, label, Icon]) => {
              const active = tab === key;
              return (
                <button
                  key={key}
                  onClick={() => setTab(key)}
                  className={`mb-1 flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm font-semibold transition ${active ? 'bg-zinc-900 text-white' : 'text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900'}`}
                >
                  <Icon size={16} /> {label}
                </button>
              );
            })}
        </div>

        <div className="xl:col-span-3 rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
          <div className="mb-4">
            <label className="mb-1 block text-xs font-bold uppercase tracking-wider text-zinc-500">dataset_id</label>
            <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)} className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm">
              {datasets.map((item) => (
                <option key={item.id} value={item.id}>{item.id} · {item.name}</option>
              ))}
            </select>
          </div>

          {tab === 'overview' && (
            <div className="space-y-4">
              <h3 className="text-lg font-bold text-zinc-900">任务列表</h3>
              <div className="max-h-72 overflow-auto rounded-xl border border-zinc-200">
                <table className="w-full text-left text-sm">
                  <thead className="sticky top-0 bg-zinc-50 text-zinc-500">
                    <tr>
                      <th className="px-3 py-2">作业ID</th>
                      <th className="px-3 py-2">类型</th>
                      <th className="px-3 py-2">状态</th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobs.map((job) => (
                      <tr
                        key={job.job_id}
                        className={`cursor-pointer border-t border-zinc-100 ${selectedJobId === job.job_id ? 'bg-zinc-100' : 'hover:bg-zinc-50'}`}
                        onClick={() => setSelectedJobId(job.job_id)}
                      >
                        <td className="px-3 py-2 font-mono text-xs text-zinc-700">{job.job_id}</td>
                        <td className="px-3 py-2 text-zinc-700">{job.name}</td>
                        <td className="px-3 py-2 text-zinc-600">{job.status}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div>
                <h4 className="mb-2 text-sm font-bold text-zinc-800">日志预览</h4>
                <pre className="h-48 overflow-auto rounded-xl border border-zinc-200 bg-zinc-950 p-3 text-xs leading-5 text-zinc-200">{jobLog || '请选择作业查看日志'}</pre>
              </div>
            </div>
          )}

          {tab === 'data' && (
            <div className="space-y-4">
              <h3 className="text-lg font-bold text-zinc-900">数据准备任务</h3>
              <input value={dataConfig} onChange={(e) => setDataConfig(e.target.value)} className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              <label className="flex items-center gap-2 text-sm text-zinc-700"><input type="checkbox" checked={downloadVideos} onChange={(e) => setDownloadVideos(e.target.checked)} /> 下载视频</label>
              <label className="flex items-center gap-2 text-sm text-zinc-700"><input type="checkbox" checked={agreeLicense} onChange={(e) => setAgreeLicense(e.target.checked)} /> 同意数据许可</label>
              <input type="number" value={videoLimit} onChange={(e) => setVideoLimit(Number(e.target.value))} className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              <Button disabled={busy} className="h-11 w-full gap-2" onClick={() => void runTask(() => createDataPrepareJob({
                dataset_id: datasetId,
                config: dataConfig,
                download_annotations: true,
                extract_annotations: true,
                download_videos: downloadVideos,
                video_limit: videoLimit,
                agree_license: agreeLicense,
                preprocess_limit: 0,
              }))}><Play size={16} /> 提交数据准备任务</Button>
            </div>
          )}

          {tab === 'extract' && (
            <div className="space-y-4">
              <h3 className="text-lg font-bold text-zinc-900">关键点提取任务</h3>
              <input value={extractConfig} onChange={(e) => setExtractConfig(e.target.value)} className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              <input value={weights} onChange={(e) => setWeights(e.target.value)} className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              <div className="grid grid-cols-2 gap-3">
                <input type="number" step="0.01" value={extractConf} onChange={(e) => setExtractConf(Number(e.target.value))} className="rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
                <input type="number" value={maxVideos} onChange={(e) => setMaxVideos(Number(e.target.value))} className="rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              </div>
              <input value={inputDir} onChange={(e) => setInputDir(e.target.value)} placeholder="可选：输入视频目录" className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              <input value={outDir} onChange={(e) => setOutDir(e.target.value)} placeholder="可选：输出目录" className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              <label className="flex items-center gap-2 text-sm text-zinc-700"><input type="checkbox" checked={recursive} onChange={(e) => setRecursive(e.target.checked)} /> 递归扫描目录</label>
              <Button disabled={busy} className="h-11 w-full gap-2" onClick={() => void runTask(() => createPoseExtractJob({
                dataset_id: datasetId,
                config: extractConfig,
                input_dir: inputDir || undefined,
                out_dir: outDir || undefined,
                recursive,
                video_ext: 'mp4',
                weights,
                conf: extractConf,
                max_videos: maxVideos,
              }))}><Play size={16} /> 提交提取任务</Button>
            </div>
          )}

          {tab === 'train' && (
            <div className="space-y-4">
              <h3 className="text-lg font-bold text-zinc-900">模型训练任务</h3>
              <input value={trainConfig} onChange={(e) => setTrainConfig(e.target.value)} className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              <input value={yolo2dDir} onChange={(e) => setYolo2dDir(e.target.value)} placeholder="可选：覆盖 yolo2d_dir" className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              <input value={gt3dDir} onChange={(e) => setGt3dDir(e.target.value)} placeholder="可选：覆盖 gt3d_dir" className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              <input value={artifactDir} onChange={(e) => setArtifactDir(e.target.value)} placeholder="可选：覆盖 artifact_dir" className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              <label className="flex items-center gap-2 text-sm text-zinc-700"><input type="checkbox" checked={exportOnnx} onChange={(e) => setExportOnnx(e.target.checked)} /> 同时导出 ONNX</label>
              <Button disabled={busy} className="h-11 w-full gap-2" onClick={() => void runTask(() => createTrainJob({
                dataset_id: datasetId,
                config: trainConfig,
                yolo2d_dir: yolo2dDir || undefined,
                gt3d_dir: gt3dDir || undefined,
                artifact_dir: artifactDir || undefined,
                export_onnx: exportOnnx,
              }))}><Play size={16} /> 提交训练任务</Button>
            </div>
          )}

          {tab === 'multiview' && (
            <div className="space-y-4">
              <h3 className="text-lg font-bold text-zinc-900">多机位处理任务</h3>
              <input value={multiviewConfig} onChange={(e) => setMultiviewConfig(e.target.value)} className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              <input type="number" value={limitSessions} onChange={(e) => setLimitSessions(Number(e.target.value))} className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              <Button disabled={busy} className="h-11 w-full gap-2" onClick={() => void runTask(() => createMultiviewJob({
                config: multiviewConfig,
                limit_sessions: limitSessions,
              }))}><Play size={16} /> 提交多机位任务</Button>
            </div>
          )}

          {tab === 'evaluate' && (
            <div className="space-y-4">
              <h3 className="text-lg font-bold text-zinc-900">评测任务</h3>
              <input value={evalInputDir} onChange={(e) => setEvalInputDir(e.target.value)} className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              <div className="grid grid-cols-2 gap-3">
                <input value={evalStyle} onChange={(e) => setEvalStyle(e.target.value)} className="rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
                <input type="number" value={evalMaxVideos} onChange={(e) => setEvalMaxVideos(Number(e.target.value))} className="rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              </div>
              <input value={evalOutput} onChange={(e) => setEvalOutput(e.target.value)} className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm" />
              <Button disabled={busy} className="h-11 w-full gap-2" onClick={() => void runTask(() => createEvaluateJob({
                dataset_id: datasetId,
                input_dir: evalInputDir,
                style: evalStyle,
                max_videos: evalMaxVideos,
                output_csv: evalOutput,
              }))}><Play size={16} /> 提交评测任务</Button>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
