import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react';
import {
  Activity,
  Database,
  FileStack,
  Film,
  LoaderCircle,
  Pause,
  Play,
  RefreshCw,
  Server,
  TriangleAlert,
} from 'lucide-react';
import { Pose2DViewport } from '../components/Pose2DViewport';
import { Pose3DViewport } from '../components/Pose3DViewport';
import { Button } from '../components/ui/Button';
import {
  createTrainJob,
  fetchArtifactManifest,
  backendBaseUrl,
  fetchArtifactStatus,
  fetchDatasets,
  fetchHealth,
  fetchJobProgress,
  fetchJobs,
  fetchPosePreview,
  fetchSourcePreview,
  fetchStandards,
  type ArtifactManifestPayload,
  type ArtifactStatus,
  type DatasetItem,
  type JobItem,
  type PosePreviewAlignment,
  type PosePreviewPayload,
  type SourcePreviewItem,
  type SourcePreviewPayload,
  type StandardItem,
} from '../lib/api';

type StepStatus = 'ready' | 'running' | 'waiting' | 'error';
type SourceGroup = {
  key: string;
  samples: SourcePreviewItem[];
  label: string;
  totalSizeBytes: number;
  generatedViews: number;
  completedViews: number;
  totalViews: number;
};

const CAMERA_TOKEN_PATTERN = /_c(\d+)_/i;
const MAX_LAYOUT_VIEW_COUNT = 6;
const VIEW_GRID_CLASSES_BY_COUNT: Record<number, string> = {
  1: 'grid grid-cols-1 gap-3',
  2: 'grid grid-cols-1 gap-3 md:grid-cols-2',
  3: 'grid grid-cols-1 gap-3 md:grid-cols-3',
  4: 'grid grid-cols-1 gap-3 md:grid-cols-2 2xl:grid-cols-4',
  5: 'grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-5',
  6: 'grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6',
};
const ALIGNMENT_CAMERA_GRID_CLASSES_BY_COUNT: Record<number, string> = {
  1: 'grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2',
  2: 'grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-2',
  3: 'grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3',
  4: 'grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-2 2xl:grid-cols-4',
  5: 'grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-5',
  6: 'grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6',
};
const THREE_D_PANEL_CLASS = 'flex h-full min-h-[0] flex-col rounded-xl border border-zinc-200 bg-stone-50 p-3 xl:sticky xl:top-4';
const TRAIN_PROGRESS_STALL_MS = 20_000;
const SYNC_DRIFT_TOLERANCE = 0.05;
const SYNC_TICK_MS = 40;
const SYNC_PAUSE_SETTLE_MS = 72;

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

function formatDecimal(value: number | undefined | null, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '-';
  }
  return Number(value).toFixed(digits);
}

function formatFrameOffset(value: number | undefined | null): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '-';
  }
  const numberValue = Math.trunc(value);
  return `${numberValue > 0 ? '+' : ''}${numberValue}f`;
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

function normalizeSequenceKey(pathValue: string): string {
  const name = pathValue.split('/').at(-1) ?? pathValue;
  return name.replace(/\.mp4$/i, '').replace(CAMERA_TOKEN_PATTERN, '_cAll_');
}

function parseCameraLabel(sample: SourcePreviewItem): string {
  if (sample.camera_id) {
    return `视角 ${sample.camera_id}`;
  }
  const matched = sample.name.match(CAMERA_TOKEN_PATTERN);
  if (!matched) {
    return '视角未知';
  }
  return `视角 c${matched[1]}`;
}

function toMediaUrl(pathValue: string, cacheKey?: string): string {
  if (!pathValue) {
    return '';
  }
  const suffix = cacheKey ? `?v=${encodeURIComponent(cacheKey)}` : '';
  return `${backendBaseUrl}${pathValue}${suffix}`;
}

function waitForVideoPlayable(video: HTMLVideoElement, timeoutMs = 4000): Promise<void> {
  if (video.readyState >= 2) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    let done = false;
    let timerId = 0;
    const cleanup = () => {
      video.removeEventListener('loadeddata', finish);
      video.removeEventListener('canplay', finish);
      video.removeEventListener('error', finish);
      window.clearTimeout(timerId);
    };
    const finish = () => {
      if (done) {
        return;
      }
      done = true;
      cleanup();
      resolve();
    };
    timerId = window.setTimeout(finish, timeoutMs);
    video.addEventListener('loadeddata', finish, { once: true });
    video.addEventListener('canplay', finish, { once: true });
    video.addEventListener('error', finish, { once: true });
  });
}

function seekVideo(video: HTMLVideoElement, timeSeconds: number): void {
  try {
    if (typeof video.fastSeek === 'function') {
      video.fastSeek(timeSeconds);
      return;
    }
  } catch {
    // fastSeek 失败时回退为 currentTime 赋值
  }
  video.currentTime = timeSeconds;
}

function pickMedian(values: number[], fallback = 0): number {
  const ordered = values.filter((value) => Number.isFinite(value)).sort((left, right) => left - right);
  if (ordered.length === 0) {
    return fallback;
  }
  const middle = Math.floor(ordered.length / 2);
  if (ordered.length % 2 === 1) {
    return ordered[middle] ?? fallback;
  }
  const left = ordered[middle - 1] ?? fallback;
  const right = ordered[middle] ?? fallback;
  return (left + right) / 2;
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
  const [posePreviewMap, setPosePreviewMap] = useState<Record<string, PosePreviewPayload>>({});
  const [posePreviewLoading, setPosePreviewLoading] = useState(false);
  const [posePreviewError, setPosePreviewError] = useState('');
  const [groupPrepareDone, setGroupPrepareDone] = useState(0);
  const [groupPrepareTotal, setGroupPrepareTotal] = useState(0);
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [selectedStandardId, setSelectedStandardId] = useState('');
  const [selectedGroupKey, setSelectedGroupKey] = useState('');
  const [summaryText, setSummaryText] = useState('');
  const [syncCurrentTime, setSyncCurrentTime] = useState(0);
  const [syncDuration, setSyncDuration] = useState(0);
  const [syncPlaying, setSyncPlaying] = useState(false);
  const [syncPlaybackRate, setSyncPlaybackRate] = useState(1);
  const [trainSubmitting, setTrainSubmitting] = useState(false);
  const [regeneratingPose, setRegeneratingPose] = useState(false);
  const [followTraining, setFollowTraining] = useState(false);
  const [followTrainJobId, setFollowTrainJobId] = useState('');
  const [followProgress, setFollowProgress] = useState(0);
  const [followCurrentStep, setFollowCurrentStep] = useState(0);
  const [followTotalStep, setFollowTotalStep] = useState(0);
  const [trainEvents, setTrainEvents] = useState<string[]>([]);
  const [trainHint, setTrainHint] = useState('');
  const [autoAdvancePending, setAutoAdvancePending] = useState(false);
  const [pendingAutoPlayJobId, setPendingAutoPlayJobId] = useState('');
  const autoPlayedJobRef = useRef('');
  const posePreviewCacheRef = useRef<Record<string, PosePreviewPayload>>({});
  const posePreviewPendingRef = useRef<Record<string, Promise<PosePreviewPayload | null>>>({});
  const previewDatasetRef = useRef('');
  const progressValueRef = useRef(0);
  const [progressUpdatedAt, setProgressUpdatedAt] = useState(0);
  const [progressWatchTs, setProgressWatchTs] = useState(0);
  const sourceVideoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const syncTickerRef = useRef<number | null>(null);
  const syncPauseSettleRef = useRef<number | null>(null);
  const syncCurrentTimeRef = useRef(0);
  const syncUiUpdateAtRef = useRef(0);
  const syncPlayingRef = useRef(false);
  const syncPauseGuardRef = useRef(false);

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
        fetchArtifactManifest(80),
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
      const preview = await fetchSourcePreview(datasetId, 160);
      setSourcePreview(preview);
      setSelectedGroupKey('');
      posePreviewCacheRef.current = {};
      posePreviewPendingRef.current = {};
      setPosePreviewMap({});
      setPosePreviewError('');
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
      if (document.visibilityState === 'hidden') {
        return;
      }
      void refreshCore();
    }, 15000);
    return () => window.clearInterval(timer);
  }, [refreshCore]);

  useEffect(() => {
    if (datasets.length === 0) {
      setSelectedDatasetId('');
      return;
    }
    const exists = datasets.some((item) => item.id === selectedDatasetId);
    if (!exists) {
      setSelectedDatasetId(datasets[0].id);
    }
  }, [datasets, selectedDatasetId]);

  useEffect(() => {
    if (standards.length === 0) {
      setSelectedStandardId('');
      return;
    }
    const exists = standards.some((item) => item.id === selectedStandardId);
    if (!exists) {
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

  const runningJobs = jobs.filter((job) => job.status === 'running').length;
  const queuedJobs = jobs.filter((job) => job.status === 'queued').length;
  const failedJobs = jobs.filter((job) => job.status === 'failed').length;

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId) ?? null,
    [datasets, selectedDatasetId],
  );
  const selectedStandard = useMemo(
    () => standards.find((item) => item.id === selectedStandardId) ?? null,
    [selectedStandardId, standards],
  );

  const orderedJobs = useMemo(
    () => [...jobs].sort((left, right) => Number(right.created_at) - Number(left.created_at)),
    [jobs],
  );
  const latestTrainJob = useMemo(
    () => orderedJobs.find((item) => item.name.includes(`train_3d_lift_${selectedDatasetId}`)) ?? null,
    [orderedJobs, selectedDatasetId],
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

  const sourceGroups = useMemo<SourceGroup[]>(() => {
    const rows = sourcePreview?.samples ?? [];
    if (rows.length === 0) {
      return [];
    }
    const groups = new Map<string, SourcePreviewItem[]>();
    for (const sample of rows) {
      const key = sample.group_key || normalizeSequenceKey(sample.path);
      const list = groups.get(key) ?? [];
      list.push(sample);
      groups.set(key, list);
    }
    return [...groups.entries()]
      .map(([key, samples]) => {
        const ordered = [...samples].sort((left, right) => left.name.localeCompare(right.name));
        const headName = ordered[0]?.name ?? key;
        const readyCount = ordered.filter((item) => item.pose2d_exists && item.pose3d_exists).length;
        const completedViews = readyCount === ordered.length ? readyCount : 0;
        return {
          key,
          samples: ordered,
          label: headName.replace(CAMERA_TOKEN_PATTERN, '_c*_'),
          totalSizeBytes: ordered.reduce((sum, item) => sum + item.size_bytes, 0),
          generatedViews: readyCount,
          completedViews,
          totalViews: ordered.length,
        };
      })
      .sort((left, right) => left.label.localeCompare(right.label));
  }, [sourcePreview]);

  useEffect(() => {
    if (sourceGroups.length === 0) {
      setSelectedGroupKey('');
      return;
    }
    const defaultGroup =
      [...sourceGroups].sort((left, right) => right.samples.length - left.samples.length)[0] ?? sourceGroups[0];
    const exists = sourceGroups.some((group) => group.key === selectedGroupKey);
    if (!exists) {
      setSelectedGroupKey(defaultGroup.key);
    }
  }, [selectedGroupKey, sourceGroups]);

  const markSourcePreviewGenerated = useCallback((samplePath: string) => {
    setSourcePreview((prev) => {
      if (!prev || prev.samples.length === 0) {
        return prev;
      }
      let changed = false;
      const nextSamples = prev.samples.map((item) => {
        if (item.path !== samplePath) {
          return item;
        }
        if (item.pose2d_exists && item.pose3d_exists) {
          return item;
        }
        changed = true;
        return {
          ...item,
          pose2d_exists: true,
          pose3d_exists: true,
        };
      });
      if (!changed) {
        return prev;
      }
      return {
        ...prev,
        samples: nextSamples,
      };
    });
  }, []);

  const currentGroup = useMemo(
    () => sourceGroups.find((group) => group.key === selectedGroupKey) ?? sourceGroups[0] ?? null,
    [selectedGroupKey, sourceGroups],
  );
  const currentGroupAllSamples = useMemo(() => currentGroup?.samples ?? [], [currentGroup]);
  const currentGroupSamples = useMemo(() => currentGroup?.samples ?? [], [currentGroup]);
  const asyncTrainGroup = useMemo(() => {
    if (sourceGroups.length === 0) {
      return null;
    }
    if (!followTraining && followProgress <= 0) {
      return null;
    }
    let ratio = followProgress;
    if (followTotalStep > 0) {
      ratio = followCurrentStep / Math.max(1, followTotalStep);
    }
    const bounded = Math.max(0, Math.min(0.999999, ratio));
    const index = Math.min(sourceGroups.length - 1, Math.floor(bounded * sourceGroups.length));
    return sourceGroups[index] ?? sourceGroups[0] ?? null;
  }, [followCurrentStep, followProgress, followTotalStep, followTraining, sourceGroups]);
  const asyncTrainGroupKey = asyncTrainGroup?.key ?? '';
  const nextGroup = useMemo(() => {
    if (!currentGroup || sourceGroups.length <= 1) {
      return null;
    }
    const currentIndex = sourceGroups.findIndex((item) => item.key === currentGroup.key);
    if (currentIndex < 0) {
      return sourceGroups[0] ?? null;
    }
    const nextIndex = (currentIndex + 1) % sourceGroups.length;
    return sourceGroups[nextIndex] ?? null;
  }, [currentGroup, sourceGroups]);

  useEffect(() => {
    previewDatasetRef.current = selectedDatasetId;
  }, [selectedDatasetId]);

  useEffect(() => {
    posePreviewCacheRef.current = posePreviewMap;
  }, [posePreviewMap]);

  useEffect(() => {
    posePreviewCacheRef.current = {};
    posePreviewPendingRef.current = {};
    setPosePreviewMap({});
    setPosePreviewError('');
  }, [selectedDatasetId]);

  const fetchPosePreviewForSample = useCallback(
    async (sample: SourcePreviewItem): Promise<PosePreviewPayload | null> => {
      const datasetId = selectedDatasetId;
      if (!datasetId) {
        return null;
      }
      const cached = posePreviewCacheRef.current[sample.path];
      if (cached) {
        return cached;
      }
      const pending = posePreviewPendingRef.current[sample.path];
      if (pending) {
        return pending;
      }

      const task = (async () => {
        let payload: PosePreviewPayload | null = null;
        for (let attempt = 0; attempt < 2; attempt += 1) {
          try {
            payload = await fetchPosePreview(datasetId, sample.path);
            break;
          } catch (err) {
            if (attempt === 1) {
              console.error(err);
            } else {
              await new Promise((resolve) => window.setTimeout(resolve, 300));
            }
          }
        }
        if (!payload) {
          return null;
        }
        if (previewDatasetRef.current !== datasetId) {
          return null;
        }
        posePreviewCacheRef.current = {
          ...posePreviewCacheRef.current,
          [sample.path]: payload,
        };
        setPosePreviewMap((prev) => {
          if (prev[sample.path]) {
            return prev;
          }
          return {
            ...prev,
            [sample.path]: payload,
          };
        });
        markSourcePreviewGenerated(sample.path);
        return payload;
      })().finally(() => {
        delete posePreviewPendingRef.current[sample.path];
      });

      posePreviewPendingRef.current[sample.path] = task;
      return task;
    },
    [markSourcePreviewGenerated, selectedDatasetId],
  );

  const ensureGroupPosePreview = useCallback(
    async (
      samples: SourcePreviewItem[],
      options: {
        showLoading?: boolean;
        updateError?: boolean;
        onProgress?: (done: number, total: number) => void;
      } = {},
    ): Promise<{ missing: string[] }> => {
      if (!selectedDatasetId || samples.length === 0) {
        return { missing: [] };
      }
      const showLoading = options.showLoading !== false;
      const updateError = options.updateError !== false;
      const total = samples.length;
      const pendingSamples = samples.filter((sample) => !posePreviewCacheRef.current[sample.path]);
      const missing: string[] = [];
      let done = total - pendingSamples.length;

      if (updateError) {
        setPosePreviewError('');
      }
      if (showLoading && pendingSamples.length > 0) {
        setPosePreviewLoading(true);
      }
      options.onProgress?.(done, total);

      if (pendingSamples.length > 0) {
        await Promise.all(
          pendingSamples.map(async (sample) => {
            const payload = await fetchPosePreviewForSample(sample);
            if (!payload) {
              missing.push(sample.path.split('/').at(-1) ?? sample.path);
            }
            done += 1;
            options.onProgress?.(done, total);
          }),
        );
      }

      if (updateError && missing.length > 0) {
        setPosePreviewError(`当前组预览未完成：${missing.join('、')}`);
      }
      if (showLoading && pendingSamples.length > 0) {
        setPosePreviewLoading(false);
      }
      return { missing };
    },
    [fetchPosePreviewForSample, selectedDatasetId],
  );

  useEffect(() => {
    if (!selectedDatasetId || currentGroupAllSamples.length === 0) {
      setPosePreviewLoading(false);
      setPosePreviewError('');
      setGroupPrepareDone(0);
      setGroupPrepareTotal(0);
      return;
    }
    let cancelled = false;

    const run = async () => {
      const result = await ensureGroupPosePreview(currentGroupAllSamples, {
        showLoading: true,
        updateError: true,
        onProgress: (done, total) => {
          if (cancelled) {
            return;
          }
          setGroupPrepareDone(done);
          setGroupPrepareTotal(total);
        },
      });
      if (cancelled) {
        return;
      }
      if (result.missing.length === 0) {
        setPosePreviewError('');
      }
    };

    void run();
    return () => {
      cancelled = true;
    };
  }, [currentGroupAllSamples, ensureGroupPosePreview, selectedDatasetId]);

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

  const curvesUrl = artifactStatus?.curves_exists ? `${backendBaseUrl}${artifactStatus.curves_url}` : '';
  const viewSlots = useMemo(
    () =>
      Array.from({ length: currentGroupSamples.length }, (_, index) => {
        const sample = currentGroupSamples[index] ?? null;
        if (!sample) {
          return {
            index,
            sample: null,
            sourceVideoUrl: '',
            pose2dVideoUrl: '',
            pose2dDataUrl: '',
            seqId: '',
            cameraLabel: `视角 ${index + 1}`,
            alignment: null,
            currentCamera: null,
          };
        }
        const payload = posePreviewMap[sample.path];
        return {
          index,
          sample,
          sourceVideoUrl: payload?.source_video_url
            ? toMediaUrl(payload.source_video_url, payload.cache_key)
            : sample.url
              ? `${backendBaseUrl}${sample.url}`
              : '',
          pose2dVideoUrl: payload?.pose2d_video_url ? toMediaUrl(payload.pose2d_video_url, payload.cache_key) : '',
          pose2dDataUrl: payload?.pose2d_data_url ? toMediaUrl(payload.pose2d_data_url, payload.cache_key) : '',
          seqId: payload?.seq_id ?? '',
          cameraLabel: parseCameraLabel(sample),
          alignment: payload?.alignment ?? null,
          currentCamera: payload?.alignment?.current_camera ?? null,
        };
      }),
    [currentGroupSamples, posePreviewMap],
  );

  const currentAlignment = useMemo<PosePreviewAlignment | null>(() => {
    for (const sample of currentGroupSamples) {
      const alignment = posePreviewMap[sample.path]?.alignment;
      if (alignment) {
        return alignment;
      }
    }
    return null;
  }, [currentGroupSamples, posePreviewMap]);

  const syncPose3dDataUrl = useMemo(() => {
    for (const sample of currentGroupSamples) {
      const payload = posePreviewMap[sample.path];
      if (payload?.pose3d_data_url) {
        return toMediaUrl(payload.pose3d_data_url, payload.cache_key);
      }
    }
    const cachedAny = Object.values(posePreviewMap).find((payload) => payload?.pose3d_data_url);
    if (cachedAny?.pose3d_data_url) {
      return toMediaUrl(cachedAny.pose3d_data_url, cachedAny.cache_key);
    }
    return '';
  }, [currentGroupSamples, posePreviewMap]);
  const activeSeqText = useMemo(() => {
    const seqSet = new Set<string>();
    for (const sample of currentGroupSamples) {
      const seqId = posePreviewMap[sample.path]?.seq_id;
      if (seqId) {
        seqSet.add(seqId);
      }
    }
    const rows = [...seqSet];
    if (rows.length === 0) {
      return '骨架序列：未就绪';
    }
    if (rows.length === 1) {
      return `骨架序列：${rows[0]}`;
    }
    return `骨架序列：${rows.length} 组`;
  }, [currentGroupSamples, posePreviewMap]);

  const activeViewSlots = viewSlots.filter((slot) => slot.sample);
  const syncReady = Boolean(
    syncPose3dDataUrl &&
    activeViewSlots.length > 0 &&
    activeViewSlots.every((slot) => slot.sourceVideoUrl && slot.pose2dDataUrl),
  );
  const layoutViewCount = Math.max(1, Math.min(activeViewSlots.length || currentGroupSamples.length || 1, MAX_LAYOUT_VIEW_COUNT));
  const viewGridClasses = VIEW_GRID_CLASSES_BY_COUNT[layoutViewCount] ?? VIEW_GRID_CLASSES_BY_COUNT[MAX_LAYOUT_VIEW_COUNT];
  const alignmentCameraGridClasses =
    ALIGNMENT_CAMERA_GRID_CLASSES_BY_COUNT[layoutViewCount] ?? ALIGNMENT_CAMERA_GRID_CLASSES_BY_COUNT[MAX_LAYOUT_VIEW_COUNT];

  const masterSourcePath = currentGroupSamples[0]?.path ?? '';
  const getMasterSourceVideo = useCallback(() => {
    if (!masterSourcePath) {
      return null;
    }
    return sourceVideoRefs.current[masterSourcePath] ?? null;
  }, [masterSourcePath]);

  const getSyncVideos = useCallback(() => {
    const nodes: HTMLVideoElement[] = [];
    for (const sample of currentGroupSamples) {
      const sourceNode = sourceVideoRefs.current[sample.path];
      if (sourceNode) {
        nodes.push(sourceNode);
      }
    }
    return nodes;
  }, [currentGroupSamples]);

  const getFollowerVideos = useCallback(() => {
    const master = getMasterSourceVideo();
    return getSyncVideos().filter((node) => node !== master);
  }, [getMasterSourceVideo, getSyncVideos]);

  const getSyncTimes = useCallback(() => {
    const master = getMasterSourceVideo();
    const masterTime = master?.currentTime;
    if (Number.isFinite(masterTime) && masterTime !== undefined) {
      return [masterTime];
    }
    return getSyncVideos()
      .filter((node) => node.readyState >= 2)
      .map((node) => node.currentTime)
      .filter((value) => Number.isFinite(value) && value >= 0);
  }, [getMasterSourceVideo, getSyncVideos]);

  const clearSyncTicker = useCallback(() => {
    if (syncTickerRef.current !== null) {
      window.clearInterval(syncTickerRef.current);
      syncTickerRef.current = null;
    }
  }, []);

  const clearSyncPauseSettleTimer = useCallback(() => {
    if (syncPauseSettleRef.current !== null) {
      window.clearTimeout(syncPauseSettleRef.current);
      syncPauseSettleRef.current = null;
    }
  }, []);

  const recomputeSyncDuration = useCallback(() => {
    const durations = getSyncVideos()
      .map((node) => node.duration)
      .filter((value) => Number.isFinite(value) && value > 0);
    if (durations.length > 0) {
      setSyncDuration(Math.min(...durations));
    }
  }, [getSyncVideos]);

  const updateSyncCurrentTime = useCallback((nextTime: number, force: boolean) => {
    const previous = syncCurrentTimeRef.current;
    syncCurrentTimeRef.current = nextTime;
    const now = window.performance.now();
    if (!force) {
      if (Math.abs(nextTime - previous) < 0.015 && now - syncUiUpdateAtRef.current < 48) {
        return;
      }
      if (now - syncUiUpdateAtRef.current < 32) {
        return;
      }
    }
    syncUiUpdateAtRef.current = now;
    setSyncCurrentTime(nextTime);
  }, []);

  const syncSeekAll = useCallback((timeSeconds: number) => {
    const normalizedTime = Math.max(0, timeSeconds);
    getSyncVideos().forEach((element) => {
      if (Math.abs(element.currentTime - normalizedTime) > 0.008) {
        seekVideo(element, normalizedTime);
      }
      if (Math.abs(element.playbackRate - syncPlaybackRate) > 0.001) {
        element.playbackRate = syncPlaybackRate;
      }
    });
    updateSyncCurrentTime(normalizedTime, true);
  }, [getSyncVideos, syncPlaybackRate, updateSyncCurrentTime]);

  const syncSetRateAll = useCallback(
    (rate: number) => {
      const master = getMasterSourceVideo();
      if (master) {
        master.playbackRate = rate;
      }
      getFollowerVideos().forEach((element) => {
        element.playbackRate = rate;
      });
    },
    [getFollowerVideos, getMasterSourceVideo],
  );

  const syncFromMaster = useCallback(
    (force: boolean) => {
      const master = getMasterSourceVideo();
      if (!master || syncPauseGuardRef.current) {
        return;
      }
      const sourceTime = master.currentTime;
      const tolerance = force ? 0.008 : SYNC_DRIFT_TOLERANCE;
      getFollowerVideos().forEach((element) => {
        if (element.readyState < 2) {
          return;
        }
        if (Math.abs(element.currentTime - sourceTime) > tolerance) {
          seekVideo(element, sourceTime);
        }
        if (Math.abs(element.playbackRate - syncPlaybackRate) > 0.001) {
          element.playbackRate = syncPlaybackRate;
        }
        if (!master.paused && element.paused) {
          void element.play().catch(() => undefined);
        }
      });
      if (Math.abs(master.playbackRate - syncPlaybackRate) > 0.001) {
        master.playbackRate = syncPlaybackRate;
      }
      updateSyncCurrentTime(sourceTime, force);
    },
    [getFollowerVideos, getMasterSourceVideo, syncPlaybackRate, updateSyncCurrentTime],
  );

  const handleSyncLoadedMetadata = useCallback(
    (video: HTMLVideoElement) => {
      video.playbackRate = syncPlaybackRate;
      const anchorTime = syncCurrentTimeRef.current;
      if (anchorTime > 0.01 && Math.abs(video.currentTime - anchorTime) > 0.02) {
        seekVideo(video, anchorTime);
      }
      if (video !== getMasterSourceVideo() && !syncPlayingRef.current && !video.paused) {
        video.pause();
      }
      recomputeSyncDuration();
    },
    [getMasterSourceVideo, recomputeSyncDuration, syncPlaybackRate],
  );

  const handleVideoLoadedData = useCallback((video: HTMLVideoElement) => {
    if (syncCurrentTimeRef.current > 0.01 || video.readyState < 2 || video.duration <= 0.05) {
      return;
    }
    if (video.currentTime > 0.001) {
      return;
    }
    try {
      video.currentTime = 0.001;
    } catch {
      return;
    }
  }, []);

  const handleSourceTimeUpdate = useCallback(() => {
    const source = getMasterSourceVideo();
    if (!source || syncPauseGuardRef.current) {
      return;
    }
    syncFromMaster(false);
  }, [getMasterSourceVideo, syncFromMaster]);

  const handleSyncPlay = useCallback(async (): Promise<boolean> => {
    if (followTraining) {
      setTrainHint('训练仍在进行，等待当前任务完成后再播放。');
      return false;
    }
    const master = getMasterSourceVideo();
    if (!master) {
      return false;
    }
    const videos = getSyncVideos();
    if (videos.length === 0) {
      return false;
    }

    clearSyncTicker();
    clearSyncPauseSettleTimer();
    syncPauseGuardRef.current = false;

    const anchorTime = pickMedian(getSyncTimes(), syncCurrentTimeRef.current);
    await Promise.all(videos.map((element) => waitForVideoPlayable(element)));
    syncSeekAll(anchorTime);
    syncSetRateAll(syncPlaybackRate);

    syncPlayingRef.current = true;
    setSyncPlaying(true);
    await Promise.allSettled(
      videos.map(async (element) => {
        if (Math.abs(element.currentTime - anchorTime) > 0.008) {
          seekVideo(element, anchorTime);
        }
        if (Math.abs(element.playbackRate - syncPlaybackRate) > 0.001) {
          element.playbackRate = syncPlaybackRate;
        }
        if (element.paused) {
          await element.play();
        }
      }),
    );

    if (master.paused) {
      syncPlayingRef.current = false;
      setSyncPlaying(false);
      return false;
    }

    syncFromMaster(true);
    return true;
  }, [
    clearSyncPauseSettleTimer,
    clearSyncTicker,
    followTraining,
    getMasterSourceVideo,
    getSyncTimes,
    getSyncVideos,
    syncFromMaster,
    syncPlaybackRate,
    syncSeekAll,
    syncSetRateAll,
  ]);

  const handleSyncPause = useCallback(() => {
    if (syncPauseGuardRef.current) {
      return;
    }
    syncPauseGuardRef.current = true;
    syncPlayingRef.current = false;
    clearSyncTicker();
    clearSyncPauseSettleTimer();

    const pauseTime = pickMedian(getSyncTimes(), syncCurrentTimeRef.current);
    const videos = getSyncVideos();
    videos.forEach((element) => {
      element.pause();
      if (Math.abs(element.playbackRate - syncPlaybackRate) > 0.001) {
        element.playbackRate = syncPlaybackRate;
      }
    });
    syncSeekAll(pauseTime);
    syncPauseSettleRef.current = window.setTimeout(() => {
      syncSeekAll(pauseTime);
      syncPauseGuardRef.current = false;
      syncPauseSettleRef.current = null;
    }, SYNC_PAUSE_SETTLE_MS);
    setSyncPlaying(false);
  }, [clearSyncPauseSettleTimer, clearSyncTicker, getSyncTimes, getSyncVideos, syncPlaybackRate, syncSeekAll]);

  const handleSyncRateChange = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) => {
      const nextRate = Number(event.target.value);
      setSyncPlaybackRate(nextRate);
      syncSetRateAll(nextRate);
    },
    [syncSetRateAll],
  );

  const handleMasterEnded = useCallback(() => {
    const master = getMasterSourceVideo();
    const endTime = master?.currentTime ?? syncDuration;
    if (endTime > 0) {
      syncSeekAll(endTime);
    }
    handleSyncPause();
    if (followTraining || !nextGroup || nextGroup.key === selectedGroupKey) {
      return;
    }
    setSelectedGroupKey(nextGroup.key);
    setAutoAdvancePending(true);
    setTrainHint(`当前素材组播放结束，切换到 ${nextGroup.label}`);
  }, [followTraining, getMasterSourceVideo, handleSyncPause, nextGroup, selectedGroupKey, syncDuration, syncSeekAll]);

  useEffect(() => {
    syncPlayingRef.current = syncPlaying;
    if (!syncPlaying) {
      clearSyncTicker();
      return undefined;
    }
    clearSyncTicker();
    syncTickerRef.current = window.setInterval(() => {
      syncFromMaster(false);
    }, SYNC_TICK_MS);
    return () => {
      clearSyncTicker();
    };
  }, [clearSyncTicker, syncFromMaster, syncPlaying]);

  useEffect(() => {
    if (!autoAdvancePending || !syncReady || followTraining) {
      return;
    }
    let cancelled = false;
    const run = async () => {
      const started = await handleSyncPlay();
      if (!cancelled && started) {
        setAutoAdvancePending(false);
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [autoAdvancePending, followTraining, handleSyncPlay, syncDuration, syncReady]);

  useEffect(() => {
    setSyncCurrentTime(0);
    setSyncDuration(0);
    setSyncPlaying(false);
    syncCurrentTimeRef.current = 0;
    syncUiUpdateAtRef.current = 0;
    syncPlayingRef.current = false;
    syncPauseGuardRef.current = false;
    sourceVideoRefs.current = {};
    clearSyncTicker();
    clearSyncPauseSettleTimer();
  }, [clearSyncPauseSettleTimer, clearSyncTicker, selectedDatasetId, selectedGroupKey]);

  useEffect(() => {
    autoPlayedJobRef.current = '';
    setPendingAutoPlayJobId('');
  }, [selectedDatasetId]);

  useEffect(() => {
    setAutoAdvancePending(false);
  }, [selectedDatasetId]);

  useEffect(() => {
    if (followTraining) {
      if (latestTrainJob) {
        setFollowTrainJobId(latestTrainJob.job_id);
      }
      return;
    }
    if (latestTrainJob?.status === 'running') {
      setFollowTraining(true);
      setFollowTrainJobId(latestTrainJob.job_id);
      if (progressUpdatedAt === 0) {
        const now = Date.now();
        setProgressUpdatedAt(now);
        setProgressWatchTs(now);
      }
      return;
    }
  }, [followTraining, latestTrainJob, progressUpdatedAt]);

  useEffect(() => {
    if (!followTraining || !followTrainJobId) {
      return undefined;
    }

    let cancelled = false;
    const readProgress = async () => {
      if (document.visibilityState === 'hidden') {
        return;
      }
      try {
        const progress = await fetchJobProgress(followTrainJobId);
        if (cancelled) {
          return;
        }
        const progressValue = Number.isFinite(progress.progress) ? progress.progress : 0;
        setFollowProgress(progressValue);
        setFollowCurrentStep(Math.max(0, Number(progress.current_step) || 0));
        setFollowTotalStep(Math.max(0, Number(progress.total_step) || 0));
        const now = Date.now();
        setProgressWatchTs(now);
        if (progressValue >= progressValueRef.current + 0.001) {
          progressValueRef.current = progressValue;
          setProgressUpdatedAt(now);
        }
        if (progress.events.length > 0) {
          const latestEvents = progress.events.slice(-4);
          setTrainEvents(latestEvents);
          setTrainHint(latestEvents[latestEvents.length - 1]);
        }
      } catch {
        // ignore transient read errors
      }
    };

    void readProgress();
    const timer = window.setInterval(() => {
      void readProgress();
    }, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [followTraining, followTrainJobId]);

  useEffect(() => {
    if (!followTraining || !followTrainJobId) {
      return;
    }
    const currentJob = jobs.find((item) => item.job_id === followTrainJobId);
    if (!currentJob) {
      return;
    }
    if (currentJob.status === 'failed') {
      setFollowTraining(false);
      setTrainHint(`训练失败：${followTrainJobId}`);
      setTrainEvents([]);
        return;
    }

    if (currentJob.status === 'succeeded') {
      setFollowProgress(1);
      setFollowCurrentStep((prev) => (followTotalStep > 0 ? followTotalStep : prev));
      setFollowTraining(false);
      setTrainHint(
        syncReady
          ? `训练完成：${followTrainJobId}，正在准备同步播放。`
          : `训练完成：${followTrainJobId}，等待骨架加载完成后可播放。`,
      );
      progressValueRef.current = 1;
      setProgressUpdatedAt(Date.now());
      setProgressWatchTs(Date.now());
      setPendingAutoPlayJobId(followTrainJobId);
      return;
    }

  }, [followTotalStep, followTrainJobId, followTraining, handleSyncPlay, jobs, syncReady, syncSeekAll]);

  const trainingStalled = useMemo(() => {
    if (!followTraining || followProgress >= 0.999) {
      return false;
    }
    if (progressUpdatedAt <= 0 || progressWatchTs <= 0) {
      return false;
    }
    return progressWatchTs - progressUpdatedAt > TRAIN_PROGRESS_STALL_MS;
  }, [followProgress, followTraining, progressUpdatedAt, progressWatchTs]);
  const progressPercent = useMemo(() => {
    const raw = Math.max(0, Math.min(100, followProgress * 100));
    if (followTraining && raw < 1) {
      return 2;
    }
    return raw;
  }, [followProgress, followTraining]);
  const progressTextPercent = useMemo(() => {
    const raw = Math.max(0, Math.min(100, followProgress * 100));
    if (followTraining && followCurrentStep > 0 && raw < 0.1) {
      return 0.1;
    }
    return raw;
  }, [followCurrentStep, followProgress, followTraining]);
  const followStepLabel = useMemo(() => {
    if (followTotalStep > 0) {
      return `${Math.min(followCurrentStep, followTotalStep)}/${followTotalStep}`;
    }
    if (followTraining) {
      return '等待批次指标';
    }
    return '-';
  }, [followCurrentStep, followTotalStep, followTraining]);

  useEffect(() => {
    if (!latestTrainJob || latestTrainJob.status !== 'succeeded') {
      return;
    }
    const shouldAutoPlay = followTrainJobId === latestTrainJob.job_id || followProgress > 0;
    if (!shouldAutoPlay) {
      return;
    }
    if (autoPlayedJobRef.current === latestTrainJob.job_id) {
      return;
    }
    setPendingAutoPlayJobId(latestTrainJob.job_id);
  }, [followProgress, followTrainJobId, latestTrainJob]);


  useEffect(() => {
    if (!pendingAutoPlayJobId || followTraining || !syncReady) {
      return;
    }
    if (autoPlayedJobRef.current === pendingAutoPlayJobId) {
      setPendingAutoPlayJobId('');
      return;
    }
    let cancelled = false;
    const run = async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 120));
      if (cancelled) {
        return;
      }
      const started = await handleSyncPlay();
      if (!started || cancelled) {
        return;
      }
      autoPlayedJobRef.current = pendingAutoPlayJobId;
      setPendingAutoPlayJobId('');
      setTrainHint(`训练完成：${pendingAutoPlayJobId}，开始同步播放当前素材组。`);
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [followTraining, handleSyncPlay, pendingAutoPlayJobId, posePreviewLoading, syncDuration, syncReady]);

  const handleStartTraining = useCallback(async () => {
    if (!selectedDatasetId) {
      return;
    }
    const trainConfigPath = selectedDataset?.train_config?.trim() || 'configs/train.yaml';
    handleSyncPause();
    syncSeekAll(0);
    setAutoAdvancePending(false);
    setTrainSubmitting(true);
    setTrainHint('');
    setTrainEvents([]);
    setPendingAutoPlayJobId('');
    autoPlayedJobRef.current = '';
    try {
      const jobId = await createTrainJob({
        dataset_id: selectedDatasetId,
        config: trainConfigPath,
        export_onnx: false,
      });
      setFollowTraining(true);
      setFollowTrainJobId(jobId);
      setFollowProgress(0);
      setFollowCurrentStep(0);
      setFollowTotalStep(0);
      progressValueRef.current = 0;
      const now = Date.now();
      setProgressUpdatedAt(now);
      setProgressWatchTs(now);
      setTrainHint(`训练任务已启动：${jobId}`);
      await refreshCore();
    } catch {
      setTrainHint('训练任务启动失败，请检查数据路径与配置。');
    } finally {
      setTrainSubmitting(false);
    }
  }, [handleSyncPause, refreshCore, selectedDataset?.train_config, selectedDatasetId, syncSeekAll]);

  const handleRegenerateCurrentGroup = useCallback(async () => {
    if (!selectedDatasetId || currentGroupAllSamples.length === 0) {
      return;
    }
    setRegeneratingPose(true);
    setPosePreviewError('');
    setGroupPrepareDone(0);
    setGroupPrepareTotal(currentGroupAllSamples.length);
    try {
      for (const sample of currentGroupAllSamples) {
        delete posePreviewCacheRef.current[sample.path];
        delete posePreviewPendingRef.current[sample.path];
      }
      setPosePreviewMap((prev) => {
        const next = { ...prev };
        for (const sample of currentGroupAllSamples) {
          delete next[sample.path];
        }
        return next;
      });

      let done = 0;
      const refreshed = await Promise.all(
        currentGroupAllSamples.map(async (sample) => {
          let payload: PosePreviewPayload | null = null;
          for (let attempt = 0; attempt < 3; attempt += 1) {
            try {
              payload = await fetchPosePreview(selectedDatasetId, sample.path, true);
              break;
            } catch {
              if (attempt < 2) {
                await new Promise((resolve) => window.setTimeout(resolve, 450));
              }
            }
          }
          done += 1;
          setGroupPrepareDone(done);
          return { path: sample.path, payload };
        }),
      );

      const updates: Record<string, PosePreviewPayload> = {};
      const failedNames: string[] = [];
      for (const item of refreshed) {
        if (item.payload) {
          updates[item.path] = item.payload;
          markSourcePreviewGenerated(item.path);
        } else {
          failedNames.push(item.path.split('/').at(-1) ?? item.path);
        }
      }

      posePreviewCacheRef.current = {
        ...posePreviewCacheRef.current,
        ...updates,
      };
      setPosePreviewMap((prev) => ({
        ...prev,
        ...updates,
      }));

      if (failedNames.length > 0) {
        setPosePreviewError(`重新解析失败：${failedNames.join('、')}`);
      } else {
        setTrainHint(`当前素材组已重新解析：${currentGroup?.label ?? ''}`);
      }
    } finally {
      setRegeneratingPose(false);
    }
  }, [currentGroup?.label, currentGroupAllSamples, markSourcePreviewGenerated, selectedDatasetId]);

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
            <Button variant="outline" onClick={() => void refreshCore()} disabled={loading} className="gap-2">
              <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
              刷新状态
            </Button>
          </div>
        </div>
      </section>

      {trainHint && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-medium text-amber-800">
          {trainHint}
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-medium text-rose-700">
          {error}
        </div>
      )}

      {(followTraining || followProgress > 0) && (
        <section className="rounded-2xl border border-zinc-200 bg-white px-5 py-4 shadow-sm">
          <div className="mb-2 flex items-center justify-between gap-3">
            <div className="text-sm font-semibold text-zinc-800">
              {followTraining ? '训练进行中' : '训练完成'}
            </div>
            <div className="text-xs font-semibold text-zinc-600">
              {progressTextPercent.toFixed(1)}% · {followStepLabel}
            </div>
          </div>
          <div className="h-2 w-full rounded-full bg-zinc-200">
            <div
              className={`h-2 rounded-full transition-all ${trainingStalled ? 'bg-amber-500' : 'bg-zinc-900'} ${
                followTraining && followProgress < 0.01 ? 'animate-pulse' : ''
              }`}
              style={{ width: `${progressPercent}%` }}
            />
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
            {trainingStalled ? (
              <span className="rounded-md border border-amber-200 bg-amber-50 px-2 py-1 text-amber-700">
                训练进度长时间未更新，建议查看日志定位卡点。
              </span>
            ) : (
              <span className="rounded-md border border-zinc-200 bg-stone-50 px-2 py-1 text-zinc-600">
                进度正常更新
              </span>
            )}
          </div>
          {trainEvents.length > 0 && (
            <div className="mt-3 rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2">
              <p className="mb-1 text-xs font-semibold text-zinc-600">训练事件</p>
              <div className="space-y-1 text-xs text-zinc-700">
                {trainEvents.map((line) => (
                  <div key={line} className="truncate">
                    {line}
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
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
                  <label htmlFor="selected-dataset" className="mb-1 block text-xs font-bold uppercase tracking-wider text-zinc-500">
                    训练数据集
                  </label>
                  <select
                    id="selected-dataset"
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
                  <label htmlFor="selected-standard" className="mb-1 block text-xs font-bold uppercase tracking-wider text-zinc-500">
                    评分标准库
                  </label>
                  <select
                    id="selected-standard"
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
              <div className="mt-3 rounded-xl border border-zinc-200 bg-stone-50 px-3 py-2">
                <p className="mb-2 text-xs font-bold uppercase tracking-wider text-zinc-500">流程状态</p>
                <div className="flex flex-wrap gap-2">
                  {pipelineSteps.map((step) => (
                    <div
                      key={step.name}
                      className="flex items-center gap-2 rounded-full border border-zinc-200 bg-white px-2.5 py-1.5"
                      title={step.detail}
                    >
                      <span className="text-xs font-semibold text-zinc-800">{step.name}</span>
                      <span
                        className={`rounded-md px-1.5 py-0.5 text-[11px] font-semibold ${
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
                      <span className="max-w-44 truncate text-[11px] text-zinc-500">{step.detail}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

          <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <h2 className="flex items-center gap-2 text-base font-bold text-zinc-800">
                <Film size={18} />
                多视角同步可视化
              </h2>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  onClick={() => void handleRegenerateCurrentGroup()}
                  disabled={regeneratingPose || !selectedDatasetId || currentGroupAllSamples.length === 0}
                  className="gap-2"
                >
                  <RefreshCw size={16} className={regeneratingPose ? 'animate-spin' : ''} />
                  重新解析当前组
                </Button>
                <Button onClick={() => void handleStartTraining()} disabled={trainSubmitting || !selectedDatasetId} className="gap-2">
                  <LoaderCircle size={16} className={trainSubmitting ? 'animate-spin' : ''} />
                  开始训练
                </Button>
              </div>
            </div>
            <div className="mb-4 rounded-xl border border-zinc-200 bg-stone-50 p-3">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <p className="text-xs font-bold uppercase tracking-wider text-zinc-500">素材组（同一动作不同 camera）</p>
                <span className="rounded-md border border-zinc-200 bg-white px-2 py-1 text-xs text-zinc-600">
                  {currentGroup ? `当前：${currentGroup.label}` : `共 ${sourceGroups.length} 组`}
                </span>
              </div>
              <div className="max-h-56 space-y-2 overflow-auto rounded-lg border border-zinc-200 bg-white p-2">
                {sourceGroups.length > 0 ? (
                  sourceGroups.map((group) => {
                    const selected = selectedGroupKey === group.key;
                    const isTraining = followTraining && asyncTrainGroupKey === group.key;
                    const statusLabel = isTraining
                      ? '训练中'
                      : group.completedViews >= group.totalViews
                        ? '已完成'
                        : '未生成';
                    const statusClass = selected
                      ? 'border border-white/40 bg-white/10 text-white'
                      : isTraining
                        ? 'border border-sky-200 bg-sky-50 text-sky-700'
                        : group.completedViews >= group.totalViews
                          ? 'border border-emerald-200 bg-emerald-50 text-emerald-700'
                          : 'border border-zinc-200 bg-white text-zinc-600';
                    return (
                      <button
                        key={group.key}
                        type="button"
                        onClick={() => setSelectedGroupKey(group.key)}
                        className={`w-full rounded-lg border px-3 py-2 text-left transition ${
                          selected
                            ? 'border-zinc-900 bg-zinc-900 text-white shadow-sm'
                            : 'border-zinc-200 bg-stone-50 text-zinc-700 hover:border-zinc-300 hover:bg-white'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <span className="min-w-0 flex-1 truncate text-sm font-semibold">{group.label}</span>
                          <span className={`rounded-md px-1.5 py-0.5 text-[11px] font-semibold ${statusClass}`}>
                            {statusLabel}
                          </span>
                          {selected ? (
                            <span className="rounded-md border border-white/40 bg-white/10 px-1.5 py-0.5 text-[11px] font-semibold text-white">
                              当前预览
                            </span>
                          ) : null}
                        </div>
                        <div className={`mt-1 text-xs ${selected ? 'text-zinc-200' : 'text-zinc-500'}`}>
                          视角数 {group.samples.length} · 骨架 {group.generatedViews}/{group.totalViews} · 体积 {formatBytes(group.totalSizeBytes)}
                        </div>
                      </button>
                    );
                  })
                ) : (
                  <div className="px-3 py-8 text-center text-sm text-zinc-500">暂无可用素材</div>
                )}
              </div>
            </div>
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <span className="rounded-md border border-zinc-200 bg-stone-50 px-2 py-1">
                  视频根目录：{sourcePreview?.video_root || '未检测到'}
                </span>
                <span className="rounded-md border border-zinc-200 bg-stone-50 px-2 py-1">
                  {syncReady ? '播放状态：就绪' : '播放状态：等待素材与骨架'}
                </span>
                <span className="rounded-md border border-zinc-200 bg-stone-50 px-2 py-1">
                  {activeSeqText}
                </span>
              </div>
            </div>

            <div className="mb-4 rounded-xl border border-zinc-200 bg-stone-50 px-4 py-3">
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  size="sm"
                  onClick={() => void handleSyncPlay()}
                  disabled={!syncReady || followTraining}
                  className="h-9 gap-2 px-3"
                >
                  <Play className="h-4 w-4" aria-hidden="true" />
                  <span>播放</span>
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleSyncPause}
                  disabled={!syncReady || followTraining}
                  className="h-9 gap-2 px-3"
                >
                  <Pause className="h-4 w-4" aria-hidden="true" />
                  <span>暂停</span>
                </Button>
                <select
                  value={syncPlaybackRate}
                  onChange={handleSyncRateChange}
                  disabled={!syncReady || followTraining}
                  className="h-9 rounded-lg border border-zinc-200 bg-white px-2 text-sm"
                >
                  <option value={0.5}>0.5x</option>
                  <option value={0.75}>0.75x</option>
                  <option value={1}>1.0x</option>
                  <option value={1.25}>1.25x</option>
                  <option value={1.5}>1.5x</option>
                </select>
                {posePreviewLoading && (
                  <span className="rounded-md border border-zinc-200 bg-white px-2 py-1 text-xs text-zinc-500">
                    骨架生成中...
                  </span>
                )}
                {followTraining && (
                  <span className="rounded-md border border-sky-200 bg-sky-50 px-2 py-1 text-xs text-sky-700">
                    训练进行中，播放锁定
                  </span>
                )}
                {!posePreviewLoading && posePreviewError && (
                  <span className="rounded-md border border-amber-200 bg-amber-50 px-2 py-1 text-xs text-amber-700">
                    {posePreviewError}
                  </span>
                )}
                <span className="ml-auto text-xs text-zinc-600">
                  {formatClock(syncCurrentTime)} / {formatClock(syncDuration)} · 训练进度 {progressTextPercent.toFixed(1)}% ·
                  {' '}
                  {followStepLabel}
                  {syncPlaying ? '（播放中）' : '（暂停）'}
                </span>
              </div>
              <div className="mt-2 flex items-center gap-2 text-xs text-zinc-600">
                <span className="rounded-md border border-zinc-200 bg-white px-2 py-1">
                  当前组预览就绪 {groupPrepareDone}/{groupPrepareTotal || currentGroupAllSamples.length}
                </span>
                <div className="h-1.5 flex-1 rounded-full bg-zinc-200">
                  <div
                    className="h-1.5 rounded-full bg-zinc-900 transition-all"
                    style={{
                      width: `${
                        (groupPrepareTotal > 0 ? (groupPrepareDone / groupPrepareTotal) * 100 : 0).toFixed(2)
                      }%`,
                    }}
                  />
                </div>
              </div>
              <input
                type="range"
                min={0}
                max={syncDuration > 0 ? syncDuration : 1}
                step={0.01}
                value={Math.min(syncCurrentTime, syncDuration > 0 ? syncDuration : syncCurrentTime)}
                onChange={(event) => syncSeekAll(Number(event.target.value))}
                disabled={!syncReady || followTraining}
                className="mt-3 h-2 w-full accent-zinc-900"
              />
            </div>

            <div className="grid gap-4 xl:grid-cols-[minmax(0,5.25fr)_minmax(320px,1.15fr)]">
              <div className="space-y-3">
                <div className={viewGridClasses}>
                  {viewSlots.map((slot, index) => (
                    <div key={`source-${slot.sample?.path || `empty-${slot.cameraLabel}`}`} className="rounded-xl border border-zinc-200 bg-stone-50 p-2.5 shadow-sm">
                      <div className="mb-1.5 space-y-1.5">
                        <h3 className="text-[13px] font-semibold leading-5 text-zinc-800">
                          素材 {index + 1} · {slot.cameraLabel}
                        </h3>
                        <div className="flex min-h-[24px] flex-wrap items-center gap-1.5">
                          {slot.currentCamera && (
                            <>
                              <span className="inline-flex rounded-md border border-zinc-200 bg-white px-1.5 py-0.5 text-[11px] text-zinc-600">
                                offset {formatFrameOffset(slot.currentCamera.offset_frames)}
                              </span>
                              <span className="inline-flex rounded-md border border-zinc-200 bg-white px-1.5 py-0.5 text-[11px] text-zinc-600">
                                trim {slot.currentCamera.trim_start}
                              </span>
                            </>
                          )}
                        </div>
                      </div>
                      {slot.sourceVideoUrl ? (
                        <video
                          key={`source-video-${slot.sourceVideoUrl || slot.sample?.path || index}`}
                          ref={(node) => {
                            if (slot.sample) {
                              sourceVideoRefs.current[slot.sample.path] = node;
                            }
                          }}
                          controls={false}
                          preload="metadata"
                          playsInline
                          muted
                          onLoadedMetadata={(event) => handleSyncLoadedMetadata(event.currentTarget)}
                          onLoadedData={(event) => handleVideoLoadedData(event.currentTarget)}
                          onTimeUpdate={() => {
                            if (index === 0) {
                              handleSourceTimeUpdate();
                            }
                          }}
                          onPlay={() => {
                            if (index === 0 && !syncPauseGuardRef.current) {
                              syncPlayingRef.current = true;
                              setSyncPlaying(true);
                              syncFromMaster(true);
                            }
                          }}
                          onPause={() => {
                            if (index === 0 && !syncPauseGuardRef.current) {
                              handleSyncPause();
                            }
                          }}
                          onEnded={() => {
                            if (index === 0) {
                              handleMasterEnded();
                            }
                          }}
                          className="aspect-video w-full rounded-lg border border-zinc-200 bg-stone-100 object-contain shadow-inner"
                        >
                          <source src={slot.sourceVideoUrl} type="video/mp4" />
                          当前浏览器无法播放视频，请检查编解码格式。
                        </video>
                      ) : (
                        <div className="flex aspect-video w-full items-center justify-center rounded-lg border border-dashed border-zinc-300 bg-white px-3 text-sm text-zinc-500">
                          {previewLoading ? '正在加载素材...' : '当前分组无该视角素材'}
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                <div className={viewGridClasses}>
                  {viewSlots.map((slot, index) => (
                    <div key={`pose2d-${slot.sample?.path || `empty-${slot.cameraLabel}`}`} className="rounded-xl border border-zinc-200 bg-stone-50 p-2.5 shadow-sm">
                      <div className="mb-1.5 space-y-1.5">
                        <h3 className="text-[13px] font-semibold leading-5 text-zinc-800">
                          2D骨架 {index + 1} · {slot.cameraLabel}
                        </h3>
                        <div className="min-h-[24px]">
                          {slot.currentCamera && (
                            <span className="inline-flex rounded-md border border-zinc-200 bg-white px-1.5 py-0.5 text-[11px] text-zinc-600">
                              误差 {formatDecimal(slot.currentCamera.sync_error_px, 1)} px
                            </span>
                          )}
                        </div>
                      </div>
                      <Pose2DViewport
                        key={slot.pose2dDataUrl || slot.sample?.path || `pose2d-${index}`}
                        dataUrl={slot.pose2dDataUrl}
                        currentTime={syncCurrentTime}
                        playing={syncPlaying}
                        videoElement={slot.sample ? sourceVideoRefs.current[slot.sample.path] ?? null : null}
                        className="aspect-video w-full"
                        emptyText={posePreviewLoading ? '正在载入 2D 预览...' : '当前视角暂无 2D 骨架'}
                      />
                    </div>
                  ))}
                </div>

              </div>

              <div className={THREE_D_PANEL_CLASS}>
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <h3 className="text-sm font-bold text-zinc-800">3D骨架（融合）</h3>
                  {followTraining && asyncTrainGroup?.label && (
                    <span className="rounded-md border border-sky-200 bg-sky-50 px-1.5 py-0.5 text-[11px] font-semibold text-sky-700">
                      训练中：{asyncTrainGroup.label}
                    </span>
                  )}
                  <span className="rounded-md border border-zinc-200 bg-white px-1.5 py-0.5 text-[11px] font-semibold text-zinc-600">
                    浏览器交互视图
                  </span>
                </div>
                <Pose3DViewport
                  key={syncPose3dDataUrl || selectedGroupKey || selectedDatasetId}
                  dataUrl={syncPose3dDataUrl}
                  currentTime={syncCurrentTime}
                  playing={syncPlaying}
                  className="min-h-[420px] flex-1 xl:min-h-0"
                  emptyText={
                    posePreviewLoading
                      ? '正在载入 3D 预览...'
                      : followTraining && asyncTrainGroup?.label
                        ? `当前分组暂无 3D 骨架，正在训练：${asyncTrainGroup.label}`
                        : '当前分组暂无 3D 骨架'
                  }
                />
              </div>
            </div>

            {currentAlignment && (
              <div className="mt-4 rounded-xl border border-zinc-200 bg-stone-50 p-3">
                <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-zinc-600">
                  <span className="rounded-md border border-zinc-200 bg-white px-2 py-1">
                    对齐模式：{currentAlignment.mode === 'aist_official_projection' ? 'AIST 官方时间轴' : currentAlignment.mode}
                  </span>
                  <span className="rounded-md border border-zinc-200 bg-white px-2 py-1">
                    相机方案：{currentAlignment.setting_name}
                  </span>
                  <span className="rounded-md border border-zinc-200 bg-white px-2 py-1">
                    公共时间轴：{formatDecimal(currentAlignment.timeline_fps, 3)} FPS · 起点 {currentAlignment.timeline_start_frame} · {currentAlignment.frame_total} 帧
                  </span>
                  <span className="rounded-md border border-zinc-200 bg-white px-2 py-1">
                    参数文件：{currentAlignment.setting_file.split('/').at(-1) || currentAlignment.setting_file}
                  </span>
                </div>
                <div className={`mt-3 ${alignmentCameraGridClasses}`}>
                  {currentAlignment.available_cameras.map((cameraId) => {
                    const geometry = currentAlignment.camera_geometry[cameraId];
                    return (
                      <div key={cameraId} className="rounded-lg border border-zinc-200 bg-white px-2.5 py-2 text-[11px] leading-5 text-zinc-600 shadow-sm">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-semibold text-zinc-900">{cameraId}</span>
                          <span className="rounded-md border border-zinc-200 bg-stone-50 px-1.5 py-0.5 text-[10px] text-zinc-600">
                            offset {formatFrameOffset(currentAlignment.camera_offsets[cameraId])}
                          </span>
                        </div>
                        <div className="mt-1.5 space-y-0.5">
                          <div>trim {currentAlignment.camera_trim_start[cameraId]} · 误差 {formatDecimal(currentAlignment.camera_sync_error_px[cameraId], 1)} px</div>
                          <div>画幅 {geometry?.image_size?.[0] ?? '-'} × {geometry?.image_size?.[1] ?? '-'}</div>
                          <div>fx/fy {formatDecimal(geometry?.focal_length_px?.[0], 1)} / {formatDecimal(geometry?.focal_length_px?.[1], 1)}</div>
                          <div>cx/cy {formatDecimal(geometry?.principal_point_px?.[0], 1)} / {formatDecimal(geometry?.principal_point_px?.[1], 1)}</div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}


            <div className="mt-4 rounded-xl border border-zinc-200 bg-stone-50 p-3">
              <h3 className="mb-2 text-sm font-bold text-zinc-800">训练曲线</h3>
              {artifactStatus?.curves_exists ? (
                <iframe
                  title="训练曲线"
                  src={curvesUrl}
                  scrolling="no"
                  className="h-[540px] w-full overflow-hidden rounded-lg border border-zinc-200 bg-white"
                />
              ) : (
                <div className="flex h-[540px] items-center justify-center rounded-lg border border-dashed border-zinc-300 bg-white text-sm text-zinc-500">
                  暂无训练曲线，请先执行训练任务
                </div>
              )}
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
