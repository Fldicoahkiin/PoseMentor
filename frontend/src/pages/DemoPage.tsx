import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react';
import {
  Activity,
  Database,
  FileStack,
  Film,
  LoaderCircle,
  RefreshCw,
  Server,
  TriangleAlert,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { Button } from '../components/ui/Button';
import {
  fetchArtifactManifest,
  backendBaseUrl,
  fetchArtifactStatus,
  fetchDatasets,
  fetchHealth,
  fetchJobs,
  fetchSourcePreview,
  fetchStandards,
  type ArtifactManifestPayload,
  type ArtifactStatus,
  type DatasetItem,
  type JobItem,
  type SourcePreviewPayload,
  type StandardItem,
} from '../lib/api';

type StepStatus = 'ready' | 'running' | 'waiting' | 'error';

function formatBytes(sizeBytes: number): string {
  if (sizeBytes < 1024) {
    return `${sizeBytes} B`;
  }
  if (sizeBytes < 1024 * 1024) {
    return `${(sizeBytes / 1024).toFixed(1)} KB`;
  }
  if (sizeBytes < 1024 * 1024 * 1024) {
    return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(sizeBytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function formatClock(totalSeconds: number): string {
  const value = Number.isFinite(totalSeconds) ? Math.max(0, totalSeconds) : 0;
  const seconds = Math.floor(value % 60);
  const minutes = Math.floor((value / 60) % 60);
  const hours = Math.floor(value / 3600);
  if (hours > 0) {
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds
      .toString()
      .padStart(2, '0')}`;
  }
  return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

function toStepStatus(jobStatus: string | undefined): StepStatus {
  if (jobStatus === 'running') {
    return 'running';
  }
  if (jobStatus === 'failed') {
    return 'error';
  }
  if (jobStatus === 'succeeded') {
    return 'ready';
  }
  return 'waiting';
}

export default function DemoPage() {
  const [loading, setLoading] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [error, setError] = useState('');
  const [health, setHealth] = useState('unknown');
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [standards, setStandards] = useState<StandardItem[]>([]);
  const [jobs, setJobs] = useState<JobItem[]>([]);
  const [artifactStatus, setArtifactStatus] = useState<ArtifactStatus | null>(null);
  const [artifactManifest, setArtifactManifest] = useState<ArtifactManifestPayload | null>(null);
  const [sourcePreview, setSourcePreview] = useState<SourcePreviewPayload | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [selectedStandardId, setSelectedStandardId] = useState('');
  const [selectedVideoIndex, setSelectedVideoIndex] = useState(0);
  const [summaryText, setSummaryText] = useState('');
  const [syncCurrentTime, setSyncCurrentTime] = useState(0);
  const [syncDuration, setSyncDuration] = useState(0);
  const [syncPlaying, setSyncPlaying] = useState(false);
  const [syncPlaybackRate, setSyncPlaybackRate] = useState(1);
  const syncDurationsRef = useRef<{ source: number; pose2d: number; pose3d: number }>({
    source: 0,
    pose2d: 0,
    pose3d: 0,
  });
  const sourceVideoRef = useRef<HTMLVideoElement | null>(null);
  const pose2dVideoRef = useRef<HTMLVideoElement | null>(null);
  const pose3dVideoRef = useRef<HTMLVideoElement | null>(null);

  const refreshCore = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [healthResp, datasetsResp, standardsResp, jobsResp, artifactsResp, manifestResp] = await Promise.all([
        fetchHealth(),
        fetchDatasets(),
        fetchStandards(),
        fetchJobs(),
        fetchArtifactStatus(),
        fetchArtifactManifest(120),
      ]);
      setHealth(healthResp.status);
      setDatasets(datasetsResp);
      setStandards(standardsResp);
      setJobs(jobsResp);
      setArtifactStatus(artifactsResp);
      setArtifactManifest(manifestResp);
    } catch (err) {
      console.error(err);
      setError('无法连接后端，请先启动 backend_api.py');
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshPreview = useCallback(async (datasetId: string) => {
    if (!datasetId) {
      return;
    }
    setPreviewLoading(true);
    try {
      const preview = await fetchSourcePreview(datasetId, 8);
      setSourcePreview(preview);
      setSelectedVideoIndex(0);
    } catch (err) {
      console.error(err);
      setError('素材预览读取失败，请检查数据目录和 dataset 配置。');
    } finally {
      setPreviewLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshCore();
    const timer = window.setInterval(() => {
      void refreshCore();
    }, 10000);
    return () => window.clearInterval(timer);
  }, [refreshCore]);

  useEffect(() => {
    if (!selectedDatasetId && datasets.length > 0) {
      setSelectedDatasetId(datasets[0].id);
    }
  }, [datasets, selectedDatasetId]);

  useEffect(() => {
    if (!selectedStandardId && standards.length > 0) {
      setSelectedStandardId(standards[0].id);
    }
  }, [selectedStandardId, standards]);

  useEffect(() => {
    if (!selectedDatasetId) {
      return;
    }
    void refreshPreview(selectedDatasetId);
  }, [refreshPreview, selectedDatasetId]);

  useEffect(() => {
    const run = async () => {
      if (!artifactStatus?.summary_exists) {
        setSummaryText('');
        return;
      }
      try {
        const response = await fetch(`${backendBaseUrl}${artifactStatus.summary_url}`);
        if (!response.ok) {
          setSummaryText('训练摘要读取失败');
          return;
        }
        const text = await response.text();
        setSummaryText(text);
      } catch {
        setSummaryText('训练摘要读取失败');
      }
    };
    void run();
  }, [artifactStatus]);

  const runningJobs = useMemo(() => jobs.filter((job) => job.status === 'running').length, [jobs]);
  const queuedJobs = useMemo(() => jobs.filter((job) => job.status === 'queued').length, [jobs]);
  const failedJobs = useMemo(() => jobs.filter((job) => job.status === 'failed').length, [jobs]);

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId) ?? null,
    [datasets, selectedDatasetId],
  );
  const selectedStandard = useMemo(
    () => standards.find((item) => item.id === selectedStandardId) ?? null,
    [selectedStandardId, standards],
  );

  const orderedJobs = useMemo(
    () => [...jobs].sort((left, right) => right.created_at.localeCompare(left.created_at)),
    [jobs],
  );

  const latestJobByKeyword = useCallback(
    (keyword: string): JobItem | null => orderedJobs.find((item) => item.name.includes(keyword)) ?? null,
    [orderedJobs],
  );

  const pipelineSteps = useMemo(() => {
    const prepareJob = latestJobByKeyword('data_prepare');
    const extractJob = latestJobByKeyword(`pose_extract_${selectedDatasetId}`);
    const trainJob = latestJobByKeyword(`train_3d_lift_${selectedDatasetId}`);
    const multiviewJob = latestJobByKeyword('multiview_prepare');

    return [
      {
        name: '素材入库检查',
        status: sourcePreview && sourcePreview.samples.length > 0 ? 'ready' : 'waiting',
        detail: sourcePreview && sourcePreview.samples.length > 0 ? `已发现 ${sourcePreview.samples.length} 个样例视频` : '待导入素材',
      },
      {
        name: '多机位对齐与格式化',
        status: selectedDataset?.mode === 'multiview' ? toStepStatus(multiviewJob?.status) : 'waiting',
        detail: selectedDataset?.mode === 'multiview' ? (multiviewJob ? multiviewJob.status : '尚未执行') : '当前非多机位数据集',
      },
      {
        name: '2D关键点提取',
        status: toStepStatus(extractJob?.status),
        detail: extractJob ? extractJob.status : '尚未执行',
      },
      {
        name: '3D模型训练',
        status: toStepStatus(trainJob?.status),
        detail: trainJob ? trainJob.status : '尚未执行',
      },
      {
        name: '产物归档',
        status: artifactManifest && artifactManifest.count > 0 ? 'ready' : 'waiting',
        detail: artifactManifest ? `已归档 ${artifactManifest.count} 个文件` : '尚无产物',
      },
      {
        name: '数据准备任务',
        status: toStepStatus(prepareJob?.status),
        detail: prepareJob ? prepareJob.status : '按需执行',
      },
    ] as { name: string; status: StepStatus; detail: string }[];
  }, [artifactManifest, latestJobByKeyword, selectedDataset?.mode, selectedDatasetId, sourcePreview]);

  const currentVideo = useMemo(
    () => sourcePreview?.samples[selectedVideoIndex] ?? null,
    [selectedVideoIndex, sourcePreview],
  );

  const modelFiles = useMemo(
    () => artifactManifest?.files.filter((item) => item.kind === 'model').slice(0, 6) ?? [],
    [artifactManifest],
  );

  const reportFiles = useMemo(
    () =>
      artifactManifest?.files
        .filter((item) => item.kind === 'report' || item.kind === 'visualization')
        .slice(0, 12) ?? [],
    [artifactManifest],
  );

  const sampleVideoUrl =
    artifactStatus?.sample_video_exists && artifactStatus.sample_video_url
      ? `${backendBaseUrl}${artifactStatus.sample_video_url}`
      : '';
  const sample2dVideoUrl =
    artifactStatus?.sample_2d_video_exists && artifactStatus.sample_2d_video_url
      ? `${backendBaseUrl}${artifactStatus.sample_2d_video_url}`
      : '';
  const sample3dVideoUrl =
    artifactStatus?.sample_3d_video_exists && artifactStatus.sample_3d_video_url
      ? `${backendBaseUrl}${artifactStatus.sample_3d_video_url}`
      : '';
  const sample2dUrl = artifactStatus?.sample_2d_exists ? `${backendBaseUrl}${artifactStatus.sample_2d_url}` : '';
  const sample3dUrl = artifactStatus?.sample_3d_exists ? `${backendBaseUrl}${artifactStatus.sample_3d_url}` : '';
  const curvesUrl = artifactStatus?.curves_exists ? `${backendBaseUrl}${artifactStatus.curves_url}` : '';
  const currentVideoUrl = currentVideo?.url ? `${backendBaseUrl}${currentVideo.url}` : '';
  const syncSourceVideoUrl = sampleVideoUrl || currentVideoUrl;
  const syncReady = Boolean(syncSourceVideoUrl && sample2dVideoUrl && sample3dVideoUrl);
  const getSyncVideos = useCallback(
    () => [sourceVideoRef.current, pose2dVideoRef.current, pose3dVideoRef.current].filter(Boolean) as HTMLVideoElement[],
    [],
  );

  const syncFromSource = useCallback(
    (force: boolean) => {
      const source = sourceVideoRef.current;
      if (!source) {
        return;
      }
      const sourceTime = source.currentTime;
      const tolerance = force ? 0.01 : 0.08;
      [pose2dVideoRef.current, pose3dVideoRef.current].forEach((element) => {
        if (!element) {
          return;
        }
        if (Math.abs(element.currentTime - sourceTime) > tolerance) {
          element.currentTime = sourceTime;
        }
        if (!source.paused && element.paused) {
          void element.play().catch(() => undefined);
        }
        if (source.paused && !element.paused) {
          element.pause();
        }
      });
      setSyncCurrentTime(sourceTime);
    },
    [],
  );

  const syncSeekAll = useCallback((timeSeconds: number) => {
    getSyncVideos().forEach((element) => {
      if (Math.abs(element.currentTime - timeSeconds) > 0.01) {
        element.currentTime = timeSeconds;
      }
    });
    setSyncCurrentTime(timeSeconds);
  }, [getSyncVideos]);

  const syncSetRateAll = useCallback(
    (rate: number) => {
      getSyncVideos().forEach((element) => {
        element.playbackRate = rate;
      });
    },
    [getSyncVideos],
  );

  const handleSyncLoadedMetadata = useCallback((role: 'source' | 'pose2d' | 'pose3d', video: HTMLVideoElement) => {
    if (Number.isFinite(video.duration) && video.duration > 0) {
      syncDurationsRef.current[role] = video.duration;
      const validDurations = Object.values(syncDurationsRef.current).filter((value) => Number.isFinite(value) && value > 0);
      if (validDurations.length > 0) {
        setSyncDuration(Math.min(...validDurations));
      }
    }
    video.playbackRate = syncPlaybackRate;
  }, [syncPlaybackRate]);

  const handleSourceTimeUpdate = useCallback(() => {
    const source = sourceVideoRef.current;
    if (!source) {
      return;
    }
    if (syncDuration <= 0 && Number.isFinite(source.duration) && source.duration > 0) {
      setSyncDuration(source.duration);
    }
    syncFromSource(false);
  }, [syncDuration, syncFromSource]);

  const handleSyncPlay = useCallback(async () => {
    const source = sourceVideoRef.current;
    if (source) {
      source.playbackRate = syncPlaybackRate;
      try {
        await source.play();
      } catch (err) {
        console.warn(err);
      }
      syncFromSource(true);
      setSyncPlaying(!source.paused);
      return;
    }

    for (const element of getSyncVideos()) {
      element.playbackRate = syncPlaybackRate;
      try {
        await element.play();
      } catch (err) {
        console.warn(err);
      }
    }
    setSyncPlaying(true);
  }, [getSyncVideos, syncFromSource, syncPlaybackRate]);

  const handleSyncPause = useCallback(() => {
    getSyncVideos().forEach((element) => {
      element.pause();
    });
    setSyncPlaying(false);
  }, [getSyncVideos]);

  const handleSyncSeekInput = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const nextTime = Number(event.target.value);
      syncSeekAll(nextTime);
    },
    [syncSeekAll],
  );

  const handleSyncRateChange = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) => {
      const nextRate = Number(event.target.value);
      setSyncPlaybackRate(nextRate);
      syncSetRateAll(nextRate);
    },
    [syncSetRateAll],
  );

  useEffect(() => {
    if (!syncPlaying) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      syncFromSource(false);
    }, 120);
    return () => window.clearInterval(timer);
  }, [syncFromSource, syncPlaying]);

  useEffect(() => {
    setSyncCurrentTime(0);
    setSyncDuration(0);
    setSyncPlaying(false);
    syncDurationsRef.current = { source: 0, pose2d: 0, pose3d: 0 };
  }, [syncSourceVideoUrl, sample2dVideoUrl, sample3dVideoUrl]);

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <section className="rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-black tracking-tight text-zinc-900">训练工作台</h1>
            <p className="mt-2 text-zinc-600">
              左侧管理素材与标准化流程，右侧查看训练视频、2D/3D骨架和训练产物。
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Link to="/admin">
              <Button variant="outline">任务控制台</Button>
            </Link>
            <Button variant="outline" onClick={() => void refreshCore()} disabled={loading} className="gap-2">
              <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
              刷新状态
            </Button>
          </div>
        </div>
      </section>

      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-medium text-rose-700">
          {error}
        </div>
      )}

      <section className="grid grid-cols-2 gap-4 xl:grid-cols-5">
        <div className="rounded-xl border border-zinc-200 bg-stone-50 p-4 shadow-sm">
          <div className="flex items-center justify-between text-zinc-500">
            <span className="text-xs font-bold uppercase tracking-wider">后端状态</span>
            <Server size={16} />
          </div>
          <div className="mt-3 text-2xl font-black text-zinc-900">{health === 'ok' ? '在线' : '离线'}</div>
        </div>

        <div className="rounded-xl border border-zinc-200 bg-stone-50 p-4 shadow-sm">
          <div className="flex items-center justify-between text-zinc-500">
            <span className="text-xs font-bold uppercase tracking-wider">数据集</span>
            <Database size={16} />
          </div>
          <div className="mt-3 text-2xl font-black text-zinc-900">{datasets.length}</div>
        </div>

        <div className="rounded-xl border border-zinc-200 bg-stone-50 p-4 shadow-sm">
          <div className="flex items-center justify-between text-zinc-500">
            <span className="text-xs font-bold uppercase tracking-wider">运行中任务</span>
            <Activity size={16} />
          </div>
          <div className="mt-3 text-2xl font-black text-zinc-900">{runningJobs}</div>
        </div>

        <div className="rounded-xl border border-zinc-200 bg-stone-50 p-4 shadow-sm">
          <div className="flex items-center justify-between text-zinc-500">
            <span className="text-xs font-bold uppercase tracking-wider">排队任务</span>
            <Activity size={16} />
          </div>
          <div className="mt-3 text-2xl font-black text-zinc-900">{queuedJobs}</div>
        </div>
        <div className="rounded-xl border border-zinc-200 bg-stone-50 p-4 shadow-sm">
          <div className="flex items-center justify-between text-zinc-500">
            <span className="text-xs font-bold uppercase tracking-wider">失败任务</span>
            <TriangleAlert size={16} />
          </div>
          <div className="mt-3 text-2xl font-black text-zinc-900">{failedJobs}</div>
        </div>
      </section>

      <section className="space-y-6">
        <div className="space-y-6">
            <div className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="flex items-center gap-2 text-base font-bold text-zinc-800">
                  <Database size={18} />
                  素材与状态面板
                </h2>
                <Button
                  variant="outline"
                  size="sm"
                  className="gap-2"
                  disabled={previewLoading || !selectedDatasetId}
                  onClick={() => void refreshPreview(selectedDatasetId)}
                >
                  <RefreshCw size={14} className={previewLoading ? 'animate-spin' : ''} />
                  刷新素材预览
                </Button>
              </div>
              <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
                <div>
                  <label className="mb-1 block text-xs font-bold uppercase tracking-wider text-zinc-500">
                    训练数据集
                  </label>
                  <select
                    value={selectedDatasetId}
                    onChange={(event) => setSelectedDatasetId(event.target.value)}
                    className="w-full rounded-xl border border-zinc-200 bg-stone-50 px-3 py-2 text-sm"
                  >
                    {datasets.map((dataset) => (
                      <option key={dataset.id} value={dataset.id}>
                        {dataset.name} · {dataset.id}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="mb-1 block text-xs font-bold uppercase tracking-wider text-zinc-500">
                    评分标准库
                  </label>
                  <select
                    value={selectedStandardId}
                    onChange={(event) => setSelectedStandardId(event.target.value)}
                    className="w-full rounded-xl border border-zinc-200 bg-stone-50 px-3 py-2 text-sm"
                  >
                    {standards.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="mt-3 grid grid-cols-1 gap-2 lg:grid-cols-3">
                <div className="rounded-xl border border-zinc-200 bg-stone-50 px-3 py-2 text-sm text-zinc-700">
                  当前数据源：<span className="font-semibold text-zinc-900">{selectedDataset?.mode || 'unknown'}</span>
                </div>
                <div className="rounded-xl border border-zinc-200 bg-stone-50 px-3 py-2 text-sm text-zinc-700">
                  当前标准库：<span className="font-semibold text-zinc-900">{selectedStandard?.name || '-'}</span>
                </div>
                <div className="rounded-xl border border-zinc-200 bg-stone-50 px-3 py-2 text-sm text-zinc-700">
                  视频根目录：<span className="font-semibold text-zinc-900">{sourcePreview?.video_root || '-'}</span>
                </div>
              </div>
              <div className="mt-3 grid grid-cols-1 gap-2 lg:grid-cols-2">
                {pipelineSteps.map((step) => (
                  <div key={step.name} className="rounded-xl border border-zinc-200 bg-stone-50 px-3 py-2">
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-sm font-semibold text-zinc-800">{step.name}</span>
                      <span
                        className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                          step.status === 'ready'
                            ? 'bg-emerald-100 text-emerald-700'
                            : step.status === 'running'
                              ? 'bg-amber-100 text-amber-700'
                              : step.status === 'error'
                                ? 'bg-rose-100 text-rose-700'
                                : 'bg-zinc-200 text-zinc-600'
                        }`}
                      >
                        {step.status}
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-zinc-500">{step.detail}</p>
                  </div>
                ))}
              </div>
            </div>

          <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <h2 className="flex items-center gap-2 text-base font-bold text-zinc-800">
                <Film size={18} />
                四联同步可视化
              </h2>
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <span className="rounded-md border border-zinc-200 bg-stone-50 px-2 py-1">
                  视频根目录：{sourcePreview?.video_root || '未检测到'}
                </span>
                <span className="rounded-md border border-zinc-200 bg-stone-50 px-2 py-1">
                  {syncReady ? '同步模式：已就绪' : '同步模式：等待训练同步视频'}
                </span>
              </div>
            </div>

            <div className="mb-4 rounded-xl border border-zinc-200 bg-stone-50 px-4 py-3">
              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm" onClick={() => void handleSyncPlay()} disabled={!syncReady} className="h-9">
                  播放
                </Button>
                <Button size="sm" variant="outline" onClick={handleSyncPause} disabled={!syncReady} className="h-9">
                  暂停
                </Button>
                <select
                  value={syncPlaybackRate}
                  onChange={handleSyncRateChange}
                  disabled={!syncReady}
                  className="h-9 rounded-lg border border-zinc-200 bg-white px-2 text-sm"
                >
                  <option value={0.5}>0.5x</option>
                  <option value={0.75}>0.75x</option>
                  <option value={1}>1.0x</option>
                  <option value={1.25}>1.25x</option>
                  <option value={1.5}>1.5x</option>
                </select>
                <span className="ml-auto text-xs text-zinc-600">
                  {formatClock(syncCurrentTime)} / {formatClock(syncDuration)}
                  {syncPlaying ? '（播放中）' : '（暂停）'}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={syncDuration > 0 ? syncDuration : 1}
                step={0.01}
                value={Math.min(syncCurrentTime, syncDuration > 0 ? syncDuration : syncCurrentTime)}
                onChange={handleSyncSeekInput}
                disabled={!syncReady}
                className="mt-3 h-2 w-full accent-zinc-900"
              />
            </div>

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <div className="rounded-xl border border-zinc-200 bg-stone-50 p-3">
                <h3 className="mb-2 text-sm font-bold text-zinc-800">素材视频</h3>
                {syncSourceVideoUrl ? (
                  <video
                    ref={sourceVideoRef}
                    controls
                    preload="metadata"
                    playsInline
                    onLoadedMetadata={(event) => handleSyncLoadedMetadata('source', event.currentTarget)}
                    onTimeUpdate={handleSourceTimeUpdate}
                    onPlay={() => {
                      setSyncPlaying(true);
                      syncFromSource(true);
                    }}
                    onPause={() => {
                      handleSyncPause();
                    }}
                    onEnded={() => {
                      handleSyncPause();
                    }}
                    className="h-[260px] w-full rounded-lg border border-zinc-200 bg-black object-contain lg:h-[34vh]"
                  >
                    <source src={syncSourceVideoUrl} type="video/mp4" />
                    当前浏览器无法播放视频，请检查编解码格式。
                  </video>
                ) : (
                  <div className="flex h-[260px] items-center justify-center rounded-lg border border-dashed border-zinc-300 bg-white text-sm text-zinc-500 lg:h-[34vh]">
                    {previewLoading ? '正在加载素材预览...' : '当前数据集暂无可预览视频'}
                  </div>
                )}
                <div className="mt-2 grid grid-cols-1 gap-2 md:grid-cols-2">
                  {(sourcePreview?.samples || []).slice(0, 6).map((sample, index) => (
                    <button
                      key={sample.path}
                      onClick={() => setSelectedVideoIndex(index)}
                      className={`rounded-lg border px-3 py-2 text-left text-xs ${
                        selectedVideoIndex === index
                          ? 'border-zinc-900 bg-zinc-900 text-white'
                          : 'border-zinc-200 bg-white text-zinc-700 hover:bg-stone-100'
                      }`}
                    >
                      <p className="truncate font-semibold">{sample.name}</p>
                      <p className={`${selectedVideoIndex === index ? 'text-zinc-300' : 'text-zinc-500'}`}>
                        {formatBytes(sample.size_bytes)}
                      </p>
                    </button>
                  ))}
                </div>
              </div>

              <div className="rounded-xl border border-zinc-200 bg-stone-50 p-3">
                <h3 className="mb-2 text-sm font-bold text-zinc-800">2D骨架</h3>
                {sample2dVideoUrl ? (
                  <video
                    ref={pose2dVideoRef}
                    controls={false}
                    preload="metadata"
                    playsInline
                    onLoadedMetadata={(event) => handleSyncLoadedMetadata('pose2d', event.currentTarget)}
                    className="h-[260px] w-full rounded-lg border border-zinc-200 bg-black object-contain lg:h-[34vh]"
                  >
                    <source src={sample2dVideoUrl} type="video/mp4" />
                    当前浏览器无法播放视频，请检查编解码格式。
                  </video>
                ) : artifactStatus?.sample_2d_exists ? (
                  <img src={sample2dUrl} alt="训练2D样例" className="h-[260px] w-full rounded-lg border border-zinc-200 bg-white object-contain lg:h-[34vh]" />
                ) : (
                  <div className="flex h-[260px] items-center justify-center rounded-lg border border-dashed border-zinc-300 bg-white text-sm text-zinc-500 lg:h-[34vh]">
                    暂无 2D 样例，请先执行训练任务
                  </div>
                )}
              </div>

              <div className="rounded-xl border border-zinc-200 bg-stone-50 p-3">
                <h3 className="mb-2 text-sm font-bold text-zinc-800">3D骨架</h3>
                {sample3dVideoUrl ? (
                  <video
                    ref={pose3dVideoRef}
                    controls={false}
                    preload="metadata"
                    playsInline
                    onLoadedMetadata={(event) => handleSyncLoadedMetadata('pose3d', event.currentTarget)}
                    className="h-[260px] w-full rounded-lg border border-zinc-200 bg-black object-contain lg:h-[34vh]"
                  >
                    <source src={sample3dVideoUrl} type="video/mp4" />
                    当前浏览器无法播放视频，请检查编解码格式。
                  </video>
                ) : artifactStatus?.sample_3d_exists ? (
                  <iframe title="训练3D样例" src={sample3dUrl} className="h-[260px] w-full rounded-lg border border-zinc-200 bg-white lg:h-[34vh]" />
                ) : (
                  <div className="flex h-[260px] items-center justify-center rounded-lg border border-dashed border-zinc-300 bg-white text-sm text-zinc-500 lg:h-[34vh]">
                    暂无 3D 样例，请先执行训练任务
                  </div>
                )}
              </div>

              <div className="rounded-xl border border-zinc-200 bg-stone-50 p-3">
                <h3 className="mb-2 text-sm font-bold text-zinc-800">训练曲线</h3>
                {artifactStatus?.curves_exists ? (
                  <iframe title="训练曲线" src={curvesUrl} className="h-[260px] w-full rounded-lg border border-zinc-200 bg-white lg:h-[34vh]" />
                ) : (
                  <div className="flex h-[260px] items-center justify-center rounded-lg border border-dashed border-zinc-300 bg-white text-sm text-zinc-500 lg:h-[34vh]">
                    暂无训练曲线，请先执行训练任务
                  </div>
                )}
              </div>
            </div>
          </div>

          <section className="grid grid-cols-1 gap-6 2xl:grid-cols-2">
            <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="flex items-center gap-2 text-base font-bold text-zinc-800">
                  <FileStack size={18} />
                  模型产物
                </h2>
                <span className="text-xs text-zinc-500">总计 {artifactManifest?.count || 0}</span>
              </div>
              <div className="mb-3 flex flex-wrap gap-2">
                {Object.entries(artifactManifest?.by_kind || {}).map(([kind, count]) => (
                  <span key={kind} className="rounded-full border border-zinc-200 bg-stone-50 px-2 py-1 text-xs text-zinc-600">
                    {kind}: {count}
                  </span>
                ))}
              </div>
              <div className="space-y-2">
                {modelFiles.length > 0 ? (
                  modelFiles.map((item) => (
                    <div key={item.path} className="flex items-center justify-between rounded-lg border border-zinc-200 bg-stone-50 px-3 py-2">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-semibold text-zinc-800">{item.name}</p>
                        <p className="text-xs text-zinc-500">{formatTime(item.updated_at)}</p>
                      </div>
                      <a
                        href={`${backendBaseUrl}${item.url}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="rounded-md border border-zinc-200 px-2 py-1 text-xs font-semibold text-zinc-700 hover:bg-zinc-100"
                      >
                        打开
                      </a>
                    </div>
                  ))
                ) : (
                  <div className="flex h-28 items-center justify-center rounded-xl border border-dashed border-zinc-300 bg-stone-50 text-sm text-zinc-500">
                    暂无模型文件
                  </div>
                )}
              </div>
            </div>

            <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="flex items-center gap-2 text-base font-bold text-zinc-800">
                  <LoaderCircle size={18} />
                  报告与摘要
                </h2>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    if (artifactStatus?.summary_exists) {
                      window.open(`${backendBaseUrl}${artifactStatus.summary_url}`, '_blank', 'noopener,noreferrer');
                    }
                  }}
                  disabled={!artifactStatus?.summary_exists}
                >
                  打开摘要
                </Button>
              </div>
              <pre className="mb-3 h-28 overflow-auto rounded-lg border border-zinc-200 bg-zinc-950 p-3 text-xs leading-5 text-zinc-200">
                {summaryText || '暂无训练摘要'}
              </pre>
              <div className="max-h-52 space-y-2 overflow-auto">
                {reportFiles.length > 0 ? (
                  reportFiles.map((item) => (
                    <div key={item.path} className="flex items-center justify-between rounded-lg border border-zinc-200 bg-stone-50 px-3 py-2">
                      <div className="min-w-0">
                        <p className="truncate text-sm text-zinc-800">{item.path}</p>
                        <p className="text-xs text-zinc-500">
                          {item.kind} · {formatBytes(item.size_bytes)}
                        </p>
                      </div>
                      <a
                        href={`${backendBaseUrl}${item.url}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="rounded-md border border-zinc-200 px-2 py-1 text-xs font-semibold text-zinc-700 hover:bg-zinc-100"
                      >
                        打开
                      </a>
                    </div>
                  ))
                ) : (
                  <div className="flex h-20 items-center justify-center rounded-xl border border-dashed border-zinc-300 bg-stone-50 text-sm text-zinc-500">
                    暂无报告文件
                  </div>
                )}
              </div>
            </div>
          </section>
        </div>
      </section>
    </div>
  );
}
