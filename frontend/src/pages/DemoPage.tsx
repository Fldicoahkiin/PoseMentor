import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Film,
  LoaderCircle,
  RefreshCw,
} from 'lucide-react';
import { AlignmentInfoPanel } from '../components/AlignmentInfoPanel';
import { ArtifactSection } from '../components/ArtifactSection';
import { PlaybackControlBar } from '../components/PlaybackControlBar';
import { TrainingProgressBar } from '../components/TrainingProgressBar';
import { SourceGroupSelector } from '../components/SourceGroupSelector';
import { WorkbenchHeader } from '../components/WorkbenchHeader';
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
  fetchModels,
  fetchPosePreview,
  fetchSourcePreview,
  fetchStandards,
  type ArtifactManifestPayload,
  type ModelItem,
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
  formatDecimal,
  formatFrameOffset,
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
  const [models, setModels] = useState<ModelItem[]>([]);
  const [selectedModel, setSelectedModel] = useState('artifacts/lift_demo.ckpt');
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
      const [healthResp, datasetsResp, standardsResp, jobsResp, artifactsResp, manifestResp, modelsResp] = await Promise.all([
        fetchHealth(),
        fetchDatasets(),
        fetchStandards(),
        fetchJobs(),
        fetchArtifactStatus(),
        fetchArtifactManifest(80),
        fetchModels(),
      ]);
      setHealth(healthResp.status);
      setDatasets(datasetsResp);
      setStandards(standardsResp);
      setJobs(jobsResp);
      setArtifactStatus(artifactsResp);
      setArtifactManifest(manifestResp);
      setModels(modelsResp);
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
  }, [setSelectedGroupKey]);

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

  const { runningJobs, failedJobs } = useMemo(() => {
    let running = 0;
    let failed = 0;
    for (const job of jobs) {
      if (job.status === 'running') running++;
      else if (job.status === 'failed') failed++;
    }
    return { runningJobs: running, failedJobs: failed };
  }, [jobs]);

  const orderedJobs = useMemo(
    () => [...jobs].sort((left, right) => Number(right.created_at) - Number(left.created_at)),
    [jobs],
  );
  const latestTrainJob = useMemo(
    () => {
      // 推理数据集不跟踪训练，避免锁住播放
      if (selectedDataset?.stage === 'inference') return null;
      return orderedJobs.find((item) => item.name.includes(`train_3d_lift_${selectedDatasetId}`)) ?? null;
    },
    [orderedJobs, selectedDataset?.stage, selectedDatasetId],
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
  }, [followTraining, getMasterSourceVideo, handleSyncPause, nextGroup, selectedGroupKey, setSelectedGroupKey, setTrainHint, syncDuration, syncSeekAll]);

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
  }, [selectedDatasetId, setPendingAutoPlayJobId]);

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
  }, [followTraining, handleSyncPlay, pendingAutoPlayJobId, posePreviewLoading, setPendingAutoPlayJobId, setTrainHint, syncDuration, syncReady]);

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
  }, [handleSyncPause, refreshCore, selectedDataset?.train_config, selectedDatasetId, setPendingAutoPlayJobId, setTrainHint, startFollowing, syncSeekAll]);

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
  }, [currentGroup?.label, currentGroupSamples, invalidateSamples, markSourcePreviewGenerated, selectedDatasetId, setGroupPrepareDone, setGroupPrepareTotal, setPosePreviewError, setPosePreviewMap, setTrainHint]);

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* header 已合并到素材面板 */}

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

      <TrainingProgressBar
        followTraining={followTraining}
        progressPercent={progressPercent}
        progressTextPercent={progressTextPercent}
        followStepLabel={followStepLabel}
        trainingStalled={trainingStalled}
        followProgress={followProgress}
        trainEvents={trainEvents}
      />

      <section className="space-y-6">
        <div className="space-y-6">
            <WorkbenchHeader
              health={health}
              loading={loading}
              previewLoading={previewLoading}
              failedJobs={failedJobs}
              runningJobs={runningJobs}
              datasets={datasets}
              standards={standards}
              selectedDatasetId={selectedDatasetId}
              selectedStandardId={selectedStandardId}
              selectedDatasetMode={selectedDataset?.mode}
              selectedStandardName={selectedStandard?.name}
              videoRoot={sourcePreview?.video_root}
              pipelineSteps={pipelineSteps}
              onRefreshCore={() => void refreshCore()}
              onRefreshPreview={() => void refreshPreview(selectedDatasetId)}
              models={models}
              selectedModel={selectedModel}
              onDatasetChange={setSelectedDatasetId}
              onStandardChange={setSelectedStandardId}
              onModelChange={setSelectedModel}
            />

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
                  2D 渲染
                </Button>
                {selectedDataset?.stage !== 'inference' && (
                  <Button
                    variant="outline"
                    onClick={() => void handleStartTraining()}
                    disabled={trainSubmitting || !selectedDatasetId}
                    className="gap-2"
                  >
                    <LoaderCircle size={16} className={trainSubmitting ? 'animate-spin' : ''} />
                    训练
                  </Button>
                )}
              </div>
            </div>
            <SourceGroupSelector
              sourceGroups={sourceGroups}
              selectedGroupKey={selectedGroupKey}
              currentGroupLabel={currentGroup?.label}
              followTraining={followTraining}
              asyncTrainGroupKey={asyncTrainGroupKey}
              onSelectGroup={setSelectedGroupKey}
            />
            <PlaybackControlBar
              syncReady={syncReady}
              followTraining={followTraining}
              syncPlaying={syncPlaying}
              syncCurrentTime={syncCurrentTime}
              syncDuration={syncDuration}
              syncPlaybackRate={syncPlaybackRate}
              posePreviewLoading={posePreviewLoading}
              posePreviewError={posePreviewError}
              progressTextPercent={progressTextPercent}
              followStepLabel={followStepLabel}
              groupPrepareDone={groupPrepareDone}
              groupPrepareTotal={groupPrepareTotal}
              currentGroupSampleCount={currentGroupSamples.length}
              videoRoot={sourcePreview?.video_root}
              activeSeqText={activeSeqText}
              onPlay={() => void handleSyncPlay()}
              onPause={handleSyncPause}
              onRateChange={handleSyncRateChange}
              onSeek={syncSeekAll}
            />

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
                        emptyText={posePreviewLoading && slot.pose2dDataUrl ? '正在载入 2D 预览...' : '点击「当前预览」生成骨架'}
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
                    posePreviewLoading && syncPose3dDataUrl
                      ? '正在载入 3D 预览...'
                      : '点击「当前预览」生成 3D 骨架'
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

          <ArtifactSection
            artifactManifest={artifactManifest}
            artifactStatus={artifactStatus}
            modelFiles={modelFiles}
            reportFiles={reportFiles}
            summaryText={summaryText}
          />
        </div>
      </section>
    </div>
  );
}
