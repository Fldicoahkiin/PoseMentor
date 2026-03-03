import { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, Database, RefreshCw, Server, Image as ImageIcon, Cuboid } from 'lucide-react';
import { Button } from '../components/ui/Button';
import {
  backendBaseUrl,
  fetchArtifactStatus,
  fetchDatasets,
  fetchHealth,
  fetchJobs,
  type ArtifactStatus,
  type DatasetItem,
  type JobItem,
} from '../lib/api';

export default function DemoPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [health, setHealth] = useState('unknown');
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [jobs, setJobs] = useState<JobItem[]>([]);
  const [artifactStatus, setArtifactStatus] = useState<ArtifactStatus | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [healthResp, datasetsResp, jobsResp, artifactsResp] = await Promise.all([
        fetchHealth(),
        fetchDatasets(),
        fetchJobs(),
        fetchArtifactStatus(),
      ]);
      setHealth(healthResp.status);
      setDatasets(datasetsResp);
      setJobs(jobsResp);
      setArtifactStatus(artifactsResp);
    } catch (err) {
      console.error(err);
      setError('无法连接后端，请先启动 backend_api.py');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => {
      void refresh();
    }, 10000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const runningJobs = useMemo(() => jobs.filter((job) => job.status === 'running').length, [jobs]);
  const queuedJobs = useMemo(() => jobs.filter((job) => job.status === 'queued').length, [jobs]);

  const sample2dUrl = artifactStatus ? `${backendBaseUrl}${artifactStatus.sample_2d_url}` : '';
  const sample3dUrl = artifactStatus ? `${backendBaseUrl}${artifactStatus.sample_3d_url}` : '';
  const curvesUrl = artifactStatus ? `${backendBaseUrl}${artifactStatus.curves_url}` : '';

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <section className="glass-card rounded-2xl border-zinc-200 bg-white p-8">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-black tracking-tight text-zinc-900">训练工作台</h1>
            <p className="mt-2 text-zinc-600">当前版本聚焦训练流程与样例骨骼可视化，不包含在线实拍评分流程。</p>
          </div>
          <Button variant="outline" onClick={() => void refresh()} disabled={loading} className="gap-2">
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
            刷新状态
          </Button>
        </div>
      </section>

      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-medium text-rose-700">
          {error}
        </div>
      )}

      <section className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm">
          <div className="flex items-center justify-between text-zinc-500">
            <span className="text-xs font-bold uppercase tracking-wider">后端状态</span>
            <Server size={16} />
          </div>
          <div className="mt-3 text-2xl font-black text-zinc-900">{health === 'ok' ? '在线' : '离线'}</div>
        </div>

        <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm">
          <div className="flex items-center justify-between text-zinc-500">
            <span className="text-xs font-bold uppercase tracking-wider">数据集</span>
            <Database size={16} />
          </div>
          <div className="mt-3 text-2xl font-black text-zinc-900">{datasets.length}</div>
        </div>

        <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm">
          <div className="flex items-center justify-between text-zinc-500">
            <span className="text-xs font-bold uppercase tracking-wider">运行中任务</span>
            <Activity size={16} />
          </div>
          <div className="mt-3 text-2xl font-black text-zinc-900">{runningJobs}</div>
        </div>

        <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm">
          <div className="flex items-center justify-between text-zinc-500">
            <span className="text-xs font-bold uppercase tracking-wider">排队任务</span>
            <Activity size={16} />
          </div>
          <div className="mt-3 text-2xl font-black text-zinc-900">{queuedJobs}</div>
        </div>
      </section>

      <section className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
          <h2 className="mb-4 flex items-center gap-2 text-base font-bold text-zinc-800">
            <ImageIcon size={18} />
            训练素材样例（2D）
          </h2>
          {artifactStatus?.sample_2d_exists ? (
            <img src={sample2dUrl} alt="训练2D样例" className="w-full rounded-xl border border-zinc-200" />
          ) : (
            <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-zinc-300 bg-zinc-50 text-sm text-zinc-500">
              还没有样例图。先执行一次训练后会自动生成。
            </div>
          )}
        </div>

        <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
          <h2 className="mb-4 flex items-center gap-2 text-base font-bold text-zinc-800">
            <Cuboid size={18} />
            生成骨骼样例（3D）
          </h2>
          {artifactStatus?.sample_3d_exists ? (
            <iframe
              title="训练3D样例"
              src={sample3dUrl}
              className="h-72 w-full rounded-xl border border-zinc-200 bg-white"
            />
          ) : (
            <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-zinc-300 bg-zinc-50 text-sm text-zinc-500">
              还没有3D样例。先执行一次训练后会自动生成。
            </div>
          )}
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h2 className="text-base font-bold text-zinc-800">训练曲线</h2>
          <Button
            variant="outline"
            className="gap-2"
            disabled={!artifactStatus?.curves_exists}
            onClick={() => {
              if (curvesUrl) {
                window.open(curvesUrl, '_blank', 'noopener,noreferrer');
              }
            }}
          >
            打开训练曲线
          </Button>
        </div>
        <p className="text-sm text-zinc-600">
          训练每个 epoch 会刷新 loss / MPJPE 曲线和样例骨骼预览，文件位于 artifacts/visualizations。
        </p>
      </section>

      <section className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-base font-bold text-zinc-800">数据集注册表</h2>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-200 text-zinc-500">
                <th className="py-2 pr-4 font-semibold">ID</th>
                <th className="py-2 pr-4 font-semibold">名称</th>
                <th className="py-2 pr-4 font-semibold">模式</th>
                <th className="py-2 pr-4 font-semibold">阶段</th>
                <th className="py-2 font-semibold">备注</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((dataset) => (
                <tr key={dataset.id} className="border-b border-zinc-100">
                  <td className="py-2 pr-4 font-mono text-xs text-zinc-700">{dataset.id}</td>
                  <td className="py-2 pr-4 text-zinc-800">{dataset.name}</td>
                  <td className="py-2 pr-4 text-zinc-600">{dataset.mode}</td>
                  <td className="py-2 pr-4 text-zinc-600">{dataset.stage}</td>
                  <td className="py-2 text-zinc-500">{dataset.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
