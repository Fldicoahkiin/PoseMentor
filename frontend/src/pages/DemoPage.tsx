import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Database,
  FileStack,
  Film,
  LoaderCircle,
  Pause,
  Play,
  RefreshCw,
} from 'lucide-react';
import { AlignmentInfoPanel } from '../components/AlignmentInfoPanel';
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
import {
  CAMERA_TOKEN_PATTERN,
  formatBytes,
  formatClock,
  formatDecimal,
  formatFrameOffset,
  formatTime,
} from '../lib/videoUtils';
import { useDatasetSelection } from '../hooks/useDatasetSelection';
import { usePosePreview } from '../hooks/usePosePreview';
import { useSyncPlayback } from '../hooks/useSyncPlayback';
import { useTrainingFollow } from '../hooks/useTrainingFollow';
import { useSourceGroups } from '../hooks/useSourceGroups';

type StepStatus = 'ready' | 'running' | 'waiting' | 'error';

const MAX_LAYOUT_VIEW_COUNT = 6;
const VIEW_GRID_CLASSES_BY_COUNT: Record<number, string> = {
  1: 'grid grid-cols-1 gap-3',
  2: 'grid grid-cols-1 gap-3 md:grid-cols-2',
  3: 'grid grid-cols-1 gap-3 md:grid-cols-3',
  4: 'grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-2 2xl:grid-cols-3',
  5: 'grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3',
  6: 'grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3',
};
const ALIGNMENT_CAMERA_GRID_CLASSES_BY_COUNT: Record<number, string> = {
  1: 'grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2',
  2: 'grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-2',
  3: 'grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3',
  4: 'grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-2 2xl:grid-cols-4',
  5: 'grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-5',
  6: 'grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6',
};
const THREE_D_PANEL_CLASS = 'flex min-h-[0] flex-col justify-center rounded-xl border border-zinc-200 bg-stone-50 p-3 xl:sticky xl:top-4 xl:max-h-[calc(100vh-8rem)]';

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
  const [summaryText, setSummaryText] = useState('');
  const [trainSubmitting, setTrainSubmitting] = useState(false);
  const [regeneratingPose, setRegeneratingPose] = useState(false);
  const [autoAdvancePending, setAutoAdvancePending] = useState(false);
  const autoPlayedJobRef = useRef('');

  const sourceGroups = useSourceGroups(sourcePreview);
  const {
    selectedDatasetId, setSelectedDatasetId,
    selectedStandardId, setSelectedStandardId,
    selectedGroupKey, setSelectedGroupKey,
    selectedDataset, selectedStandard,
  } = useDatasetSelection(datasets, standards, sourceGroups);

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
      // pose preview 缓存在 selectedDatasetId 变化时由 usePosePreview 自动清空
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

  const { runningJobs, queuedJobs, failedJobs } = useMemo(() => {
    let running = 0;
    let queued = 0;
    let failed = 0;
    for (const job of jobs) {
      if (job.status === 'running') running++;
      else if (job.status === 'queued') queued++;
      else if (job.status === 'failed') failed++;
    }
    return { runningJobs: running, queuedJobs: queued, failedJobs: failed };
  }, [jobs]);

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
  const currentGroupSamples = useMemo(() => currentGroup?.samples ?? [], [currentGroup]);
  const currentGroupSamplePaths = useMemo(() => currentGroupSamples.map((s) => s.path), [currentGroupSamples]);
  const {
    syncPlaying, syncCurrentTime, syncDuration, syncPlaybackRate,
    sourceVideoRefs, syncPlayingRef, syncPauseGuardRef,
    setSyncPlaying,
    getMasterSourceVideo, syncSeekAll, syncFromMaster,
    handleSyncPlay: rawSyncPlay, handleSyncPause, handleSyncRateChange,
    handleSyncLoadedMetadata, handleVideoLoadedData, handleSourceTimeUpdate,
    resetSyncState,
  } = useSyncPlayback(currentGroupSamplePaths);
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

  const {
    posePreviewMap,
    posePreviewLoading,
    setPosePreviewLoading,
    posePreviewError,
    setPosePreviewError,
    groupPrepareDone,
    setGroupPrepareDone,
    groupPrepareTotal,
    setGroupPrepareTotal,
    ensureGroupPosePreview,
    invalidateSamples,
    setPosePreviewMap,
  } = usePosePreview(selectedDatasetId, markSourcePreviewGenerated);

  useEffect(() => {
    if (!selectedDatasetId || currentGroupSamples.length === 0) {
      setPosePreviewLoading(false);
      setPosePreviewError('');
      setGroupPrepareDone(0);
      setGroupPrepareTotal(0);
      return;
    }
    let cancelled = false;

    const run = async () => {
      const result = await ensureGroupPosePreview(currentGroupSamples, {
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
  }, [currentGroupSamples, ensureGroupPosePreview, selectedDatasetId, setGroupPrepareDone, setGroupPrepareTotal, setPosePreviewError, setPosePreviewLoading]);

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

  const {
    followTraining,
    followProgress, followCurrentStep, followTotalStep,
    trainHint, setTrainHint,
    trainEvents,
    trainingStalled,
    progressPercent, progressTextPercent,
    followStepLabel,
    pendingAutoPlayJobId, setPendingAutoPlayJobId,
    startFollowing,
  } = useTrainingFollow(jobs, latestTrainJob, syncReady);

  const handleSyncPlay = useCallback(async (): Promise<boolean> => {
    if (followTraining) {
      setTrainHint('训练仍在进行，等待当前任务完成后再播放。');
      return false;
    }
    return rawSyncPlay();
  }, [followTraining, rawSyncPlay, setTrainHint]);

  const asyncTrainGroup = useMemo(() => {
    if (sourceGroups.length === 0) return null;
    if (!followTraining && followProgress <= 0) return null;
    let ratio = followProgress;
    if (followTotalStep > 0) {
      ratio = followCurrentStep / Math.max(1, followTotalStep);
    }
    const bounded = Math.max(0, Math.min(0.999999, ratio));
    const index = Math.min(sourceGroups.length - 1, Math.floor(bounded * sourceGroups.length));
    return sourceGroups[index] ?? sourceGroups[0] ?? null;
  }, [followCurrentStep, followProgress, followTotalStep, followTraining, sourceGroups]);
  const asyncTrainGroupKey = asyncTrainGroup?.key ?? '';

  const layoutViewCount = Math.max(1, Math.min(activeViewSlots.length || currentGroupSamples.length || 1, MAX_LAYOUT_VIEW_COUNT));
  const viewGridClasses = VIEW_GRID_CLASSES_BY_COUNT[layoutViewCount] ?? VIEW_GRID_CLASSES_BY_COUNT[MAX_LAYOUT_VIEW_COUNT];
  const alignmentCameraGridClasses =
    ALIGNMENT_CAMERA_GRID_CLASSES_BY_COUNT[layoutViewCount] ?? ALIGNMENT_CAMERA_GRID_CLASSES_BY_COUNT[MAX_LAYOUT_VIEW_COUNT];

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

  // sync ticker effect 已迁移到 useSyncPlayback hook

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
    resetSyncState();
  }, [resetSyncState, selectedDatasetId, selectedGroupKey]);

  useEffect(() => {
    autoPlayedJobRef.current = '';
    setPendingAutoPlayJobId('');
  }, [selectedDatasetId]);

  useEffect(() => {
    setAutoAdvancePending(false);
  }, [selectedDatasetId]);

  // training follow 的 3 个 effect + 4 个 memo + autoPlay 检测由 useTrainingFollow hook 管理


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
    setPendingAutoPlayJobId('');
    autoPlayedJobRef.current = '';
    try {
      const jobId = await createTrainJob({
        dataset_id: selectedDatasetId,
        config: trainConfigPath,
        export_onnx: false,
      });
      startFollowing(jobId);
      setTrainHint(`训练任务已启动：${jobId}`);
      await refreshCore();
    } catch {
      setTrainHint('训练任务启动失败，请检查数据路径与配置。');
    } finally {
      setTrainSubmitting(false);
    }
  }, [handleSyncPause, refreshCore, selectedDataset?.train_config, selectedDatasetId, syncSeekAll]);

  const handleRegenerateCurrentGroup = useCallback(async () => {
    if (!selectedDatasetId || currentGroupSamples.length === 0) {
      return;
    }
    setRegeneratingPose(true);
    setPosePreviewError('');
    setGroupPrepareDone(0);
    setGroupPrepareTotal(currentGroupSamples.length);
    try {
      invalidateSamples(currentGroupSamples.map((s) => s.path));

      let done = 0;
      const refreshed = await Promise.all(
        currentGroupSamples.map(async (sample) => {
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

      setPosePreviewMap((prev) => ({ ...prev, ...updates }));

      if (failedNames.length > 0) {
        setPosePreviewError(`重新解析失败：${failedNames.join('、')}`);
      } else {
        setTrainHint(`当前素材组已重新解析：${currentGroup?.label ?? ''}`);
      }
    } finally {
      setRegeneratingPose(false);
    }
  }, [currentGroup?.label, currentGroupSamples, markSourcePreviewGenerated, selectedDatasetId]);

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

      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-xs shadow-sm">
        <span className="flex items-center gap-1.5 font-semibold text-zinc-700">
          <span className={`inline-block h-2 w-2 rounded-full ${health === 'ok' ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
          {health === 'ok' ? '后端在线' : '后端离线'}
        </span>
        <span className="text-zinc-300">|</span>
        <span className="text-zinc-600">数据集 <b className="text-zinc-900">{datasets.length}</b></span>
        {runningJobs > 0 && (
          <>
            <span className="text-zinc-300">|</span>
            <span className="text-amber-600">运行中 <b>{runningJobs}</b></span>
          </>
        )}
        {queuedJobs > 0 && (
          <span className="text-zinc-500">排队 <b>{queuedJobs}</b></span>
        )}
        {failedJobs > 0 && (
          <>
            <span className="text-zinc-300">|</span>
            <span className="text-rose-600">失败 <b>{failedJobs}</b></span>
          </>
        )}
      </div>

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
              <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-zinc-500">
                <span>模式 <b className="text-zinc-800">{selectedDataset?.mode || '-'}</b></span>
                <span>标准 <b className="text-zinc-800">{selectedStandard?.name || '-'}</b></span>
                <span>路径 <b className="text-zinc-800">{sourcePreview?.video_root || '-'}</b></span>
              </div>
              <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1">
                  {pipelineSteps.map((step) => (
                    <span
                      key={step.name}
                      className="flex items-center gap-1 text-[11px] text-zinc-600"
                      title={step.detail}
                    >
                      <span className={`inline-block h-1.5 w-1.5 rounded-full ${
                        step.status === 'ready'
                          ? 'bg-emerald-500'
                          : step.status === 'running'
                            ? 'bg-amber-500 animate-pulse'
                            : step.status === 'error'
                              ? 'bg-rose-500'
                              : 'bg-zinc-300'
                      }`} />
                      {step.name}
                    </span>
                  ))}
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
                  disabled={regeneratingPose || !selectedDatasetId || currentGroupSamples.length === 0}
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
                  当前组预览就绪 {groupPrepareDone}/{groupPrepareTotal || currentGroupSamples.length}
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
                    <div key={`source-${slot.sample?.path || `empty-${slot.cameraLabel}`}`} className="overflow-hidden rounded-lg border border-zinc-200 bg-stone-50 shadow-sm">
                      <div className="flex items-center gap-1.5 px-2 py-1">
                        <span className="text-[11px] font-semibold text-zinc-700">
                          {slot.cameraLabel}
                        </span>
                        {slot.currentCamera && (
                          <>
                            <span className="text-[10px] text-zinc-400">
                              {formatFrameOffset(slot.currentCamera.offset_frames)}
                            </span>
                            <span className="text-[10px] text-zinc-400">
                              t{slot.currentCamera.trim_start}
                            </span>
                          </>
                        )}
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
                          className="aspect-video max-h-[360px] w-full rounded-lg border border-zinc-200 bg-stone-100 object-contain shadow-inner"
                        >
                          <source src={slot.sourceVideoUrl} type="video/mp4" />
                          当前浏览器无法播放视频，请检查编解码格式。
                        </video>
                      ) : (
                        <div className="flex aspect-video max-h-[360px] w-full items-center justify-center rounded-lg border border-dashed border-zinc-300 bg-white px-3 text-sm text-zinc-500">
                          {previewLoading ? '正在加载素材...' : '当前分组无该视角素材'}
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                <div className={viewGridClasses}>
                  {viewSlots.map((slot, index) => (
                    <div key={`pose2d-${slot.sample?.path || `empty-${slot.cameraLabel}`}`} className="overflow-hidden rounded-lg border border-zinc-200 bg-stone-50 shadow-sm">
                      <div className="flex items-center gap-1.5 px-2 py-1">
                        <span className="text-[11px] font-semibold text-zinc-700">
                          2D · {slot.cameraLabel}
                        </span>
                        {slot.currentCamera && (
                          <span className="text-[10px] text-zinc-400">
                            {formatDecimal(slot.currentCamera.sync_error_px, 1)}px
                          </span>
                        )}
                      </div>
                      <Pose2DViewport
                        key={slot.pose2dDataUrl || slot.sample?.path || `pose2d-${index}`}
                        dataUrl={slot.pose2dDataUrl}
                        currentTime={syncCurrentTime}
                        playing={syncPlaying}
                        videoElement={slot.sample ? sourceVideoRefs.current[slot.sample.path] ?? null : null}
                        className="aspect-video max-h-[320px] w-full"
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
                  className="min-h-[320px] max-h-[480px] flex-1 xl:min-h-0"
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
              <AlignmentInfoPanel alignment={currentAlignment} gridClasses={alignmentCameraGridClasses} />
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
