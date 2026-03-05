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
  completedViews: number;
  totalViews: number;
};

const CAMERA_TOKEN_PATTERN = /_c(\d+)_/i;
const SOURCE_COLUMN_CLASSES = ['xl:col-start-1 xl:row-start-1', 'xl:col-start-2 xl:row-start-1', 'xl:col-start-3 xl:row-start-1'];
const POSE2D_COLUMN_CLASSES = ['xl:col-start-1 xl:row-start-2', 'xl:col-start-2 xl:row-start-2', 'xl:col-start-3 xl:row-start-2'];
const TRAIN_PROGRESS_STALL_MS = 20_000;

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
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [selectedStandardId, setSelectedStandardId] = useState('');
  const [selectedGroupKey, setSelectedGroupKey] = useState('');
  const [selectedGroupKeys, setSelectedGroupKeys] = useState<string[]>([]);
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
  const [prefetchHint, setPrefetchHint] = useState('');
  const [autoAdvancePending, setAutoAdvancePending] = useState(false);
  const autoPlayedJobRef = useRef('');
  const autoPlayStartedRef = useRef('');
  const posePreviewCacheRef = useRef<Record<string, PosePreviewPayload>>({});
  const posePreviewPendingRef = useRef<Record<string, Promise<PosePreviewPayload | null>>>({});
  const previewDatasetRef = useRef('');
  const progressValueRef = useRef(0);
  const [progressUpdatedAt, setProgressUpdatedAt] = useState(0);
  const [progressWatchTs, setProgressWatchTs] = useState(0);
  const sourceVideoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const pose2dVideoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const pose3dVideoRef = useRef<HTMLVideoElement | null>(null);
  const syncTickerRef = useRef<number | null>(null);
  const syncCurrentTimeRef = useRef(0);
  const syncUiUpdateAtRef = useRef(0);

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
      setSelectedGroupKeys([]);
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
        const completedViews = ordered.filter((item) => item.pose2d_exists && item.pose3d_exists).length;
        return {
          key,
          samples: ordered,
          label: headName.replace(CAMERA_TOKEN_PATTERN, '_c*_'),
          totalSizeBytes: ordered.reduce((sum, item) => sum + item.size_bytes, 0),
          completedViews,
          totalViews: ordered.length,
        };
      })
      .sort((left, right) => left.label.localeCompare(right.label));
  }, [sourcePreview]);

  useEffect(() => {
    if (sourceGroups.length === 0) {
      setSelectedGroupKey('');
      setSelectedGroupKeys([]);
      return;
    }
    const defaultGroup =
      [...sourceGroups].sort((left, right) => right.samples.length - left.samples.length)[0] ?? sourceGroups[0];
    const available = new Set(sourceGroups.map((group) => group.key));
    setSelectedGroupKeys((prev) => {
      const next = prev.filter((key) => available.has(key));
      if (next.length > 0) {
        return next;
      }
      return [defaultGroup.key];
    });
  }, [sourceGroups]);

  useEffect(() => {
    if (selectedGroupKeys.length === 0) {
      setSelectedGroupKey('');
      return;
    }
    if (!selectedGroupKeys.includes(selectedGroupKey)) {
      setSelectedGroupKey(selectedGroupKeys[0]);
    }
  }, [selectedGroupKey, selectedGroupKeys]);

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

  const toggleGroupSelected = useCallback((groupKey: string) => {
    setSelectedGroupKeys((prev) => {
      if (prev.includes(groupKey)) {
        if (prev.length === 1) {
          return prev;
        }
        return prev.filter((key) => key !== groupKey);
      }
      return [...prev, groupKey];
    });
  }, []);

  const selectAllGroups = useCallback(() => {
    setSelectedGroupKeys(sourceGroups.map((group) => group.key));
  }, [sourceGroups]);

  const playbackGroups = useMemo(() => {
    if (selectedGroupKeys.length === 0) {
      return sourceGroups;
    }
    const selectedSet = new Set(selectedGroupKeys);
    return sourceGroups.filter((group) => selectedSet.has(group.key));
  }, [selectedGroupKeys, sourceGroups]);

  const currentGroup = useMemo(
    () => playbackGroups.find((group) => group.key === selectedGroupKey) ?? playbackGroups[0] ?? null,
    [playbackGroups, selectedGroupKey],
  );
  const currentGroupSamples = useMemo(
    () => (currentGroup?.samples ?? []).slice(0, 3),
    [currentGroup],
  );
  const asyncTrainGroup = useMemo(() => {
    if (playbackGroups.length === 0) {
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
    const index = Math.min(playbackGroups.length - 1, Math.floor(bounded * playbackGroups.length));
    return playbackGroups[index] ?? playbackGroups[0] ?? null;
  }, [followCurrentStep, followProgress, followTotalStep, followTraining, playbackGroups]);
  const asyncTrainGroupKey = asyncTrainGroup?.key ?? '';
  const nextGroup = useMemo(() => {
    if (!currentGroup || playbackGroups.length <= 1) {
      return null;
    }
    const currentIndex = playbackGroups.findIndex((item) => item.key === currentGroup.key);
    if (currentIndex < 0) {
      return playbackGroups[0] ?? null;
    }
    const nextIndex = (currentIndex + 1) % playbackGroups.length;
    return playbackGroups[nextIndex] ?? null;
  }, [currentGroup, playbackGroups]);

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

      const task = fetchPosePreview(datasetId, sample.path)
        .then((payload) => {
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
        })
        .catch((err) => {
          console.error(err);
          return null;
        })
        .finally(() => {
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
      } = {},
    ): Promise<{ missing: string[] }> => {
      if (!selectedDatasetId || samples.length === 0) {
        return { missing: [] };
      }
      const showLoading = options.showLoading !== false;
      const updateError = options.updateError !== false;
      if (showLoading) {
        setPosePreviewLoading(true);
      }
      if (updateError) {
        setPosePreviewError('');
      }

      const missing: string[] = [];
      await Promise.all(
        samples.map(async (sample) => {
          const payload = await fetchPosePreviewForSample(sample);
          if (!payload) {
            missing.push(sample.path.split('/').at(-1) ?? sample.path);
          }
        }),
      );

      if (updateError && missing.length > 0) {
        setPosePreviewError(`部分视角未生成骨架：${missing.join('、')}`);
      }
      if (showLoading) {
        setPosePreviewLoading(false);
      }
      return { missing };
    },
    [fetchPosePreviewForSample, selectedDatasetId],
  );

  useEffect(() => {
    if (!selectedDatasetId || currentGroupSamples.length === 0) {
      setPosePreviewLoading(false);
      setPosePreviewError('');
      return;
    }
    let cancelled = false;

    const run = async () => {
      const result = await ensureGroupPosePreview(currentGroupSamples, {
        showLoading: true,
        updateError: true,
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
  }, [currentGroupSamples, ensureGroupPosePreview, selectedDatasetId]);

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
      [0, 1, 2].map((index) => {
        const sample = currentGroupSamples[index] ?? null;
        if (!sample) {
          return {
            index,
            sample: null,
            sourceVideoUrl: '',
            pose2dVideoUrl: '',
            seqId: '',
            cameraLabel: `视角 ${index + 1}`,
          };
        }
        const payload = posePreviewMap[sample.path];
        return {
          index,
          sample,
          sourceVideoUrl: sample.url ? `${backendBaseUrl}${sample.url}` : '',
          pose2dVideoUrl: payload?.pose2d_video_url ? `${backendBaseUrl}${payload.pose2d_video_url}` : '',
          seqId: payload?.seq_id ?? '',
          cameraLabel: parseCameraLabel(sample),
        };
      }),
    [currentGroupSamples, posePreviewMap],
  );

  const syncPose3dVideoUrl = useMemo(() => {
    for (const sample of currentGroupSamples) {
      const payload = posePreviewMap[sample.path];
      if (payload?.pose3d_video_url) {
        return `${backendBaseUrl}${payload.pose3d_video_url}`;
      }
    }
    if (artifactStatus?.sample_3d_video_exists && artifactStatus.sample_3d_video_url) {
      return `${backendBaseUrl}${artifactStatus.sample_3d_video_url}`;
    }
    return '';
  }, [artifactStatus?.sample_3d_video_exists, artifactStatus?.sample_3d_video_url, currentGroupSamples, posePreviewMap]);
  const syncPose3dIsFallback = useMemo(() => {
    if (!syncPose3dVideoUrl) {
      return false;
    }
    return Boolean(artifactStatus?.sample_3d_video_exists && syncPose3dVideoUrl.endsWith("sample_3d_latest.mp4"));
  }, [artifactStatus?.sample_3d_video_exists, syncPose3dVideoUrl]);
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
    syncPose3dVideoUrl &&
    activeViewSlots.length > 0 &&
    activeViewSlots.every((slot) => slot.sourceVideoUrl && slot.pose2dVideoUrl),
  );

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
      const pose2dNode = pose2dVideoRefs.current[sample.path];
      if (pose2dNode) {
        nodes.push(pose2dNode);
      }
    }
    if (pose3dVideoRef.current) {
      nodes.push(pose3dVideoRef.current);
    }
    return nodes;
  }, [currentGroupSamples]);

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
      if (Math.abs(nextTime - previous) < 0.09 && now - syncUiUpdateAtRef.current < 180) {
        return;
      }
      if (now - syncUiUpdateAtRef.current < 140) {
        return;
      }
    }
    syncUiUpdateAtRef.current = now;
    setSyncCurrentTime(nextTime);
  }, []);

  const syncFromMaster = useCallback(
    (force: boolean) => {
      const master = getMasterSourceVideo();
      if (!master) {
        return;
      }
      const sourceTime = master.currentTime;
      const tolerance = force ? 0.008 : 0.045;
      getSyncVideos().forEach((element) => {
        if (element === master) {
          return;
        }
        if (element.readyState < 2) {
          return;
        }
        if (Math.abs(element.currentTime - sourceTime) > tolerance) {
          if (typeof element.fastSeek === 'function') {
            element.fastSeek(sourceTime);
          } else {
            element.currentTime = sourceTime;
          }
        }
      });
      updateSyncCurrentTime(sourceTime, force);
    },
    [getMasterSourceVideo, getSyncVideos, updateSyncCurrentTime],
  );

  const syncSeekAll = useCallback((timeSeconds: number) => {
    getSyncVideos().forEach((element) => {
      if (Math.abs(element.currentTime - timeSeconds) > 0.01) {
        if (typeof element.fastSeek === 'function') {
          element.fastSeek(timeSeconds);
        } else {
          element.currentTime = timeSeconds;
        }
      }
    });
    updateSyncCurrentTime(timeSeconds, true);
  }, [getSyncVideos, updateSyncCurrentTime]);

  const syncSetRateAll = useCallback(
    (rate: number) => {
      getSyncVideos().forEach((element) => {
        element.playbackRate = rate;
      });
    },
    [getSyncVideos],
  );

  const handleSyncLoadedMetadata = useCallback(
    (video: HTMLVideoElement) => {
      video.playbackRate = syncPlaybackRate;
      const anchorTime = syncCurrentTimeRef.current;
      if (anchorTime > 0.01 && Math.abs(video.currentTime - anchorTime) > 0.02) {
        video.currentTime = anchorTime;
      }
      const master = getMasterSourceVideo();
      if (syncPlaying && master) {
        if (master === video && video.paused) {
          void video.play().catch(() => undefined);
        } else if (!video.paused) {
          video.pause();
        }
      }
      recomputeSyncDuration();
    },
    [getMasterSourceVideo, recomputeSyncDuration, syncPlaybackRate, syncPlaying],
  );

  const handleSourceTimeUpdate = useCallback(() => {
    const source = getMasterSourceVideo();
    if (!source) {
      return;
    }
    recomputeSyncDuration();
    syncFromMaster(false);
  }, [getMasterSourceVideo, recomputeSyncDuration, syncFromMaster]);

  const handleSyncPlay = useCallback(async () => {
    const master = getMasterSourceVideo();
    if (!master) {
      return;
    }
    const videos = getSyncVideos();
    for (const element of videos) {
      element.playbackRate = syncPlaybackRate;
      if (element !== master && !element.paused) {
        element.pause();
      }
    }
    try {
      await master.play();
    } catch (err) {
      console.warn(err);
      setSyncPlaying(false);
      return;
    }
    syncFromMaster(true);
    setSyncPlaying(true);
  }, [getMasterSourceVideo, getSyncVideos, syncFromMaster, syncPlaybackRate]);

  const handleSyncPause = useCallback(() => {
    getSyncVideos().forEach((element) => {
      element.pause();
    });
    if (syncTickerRef.current !== null) {
      window.clearInterval(syncTickerRef.current);
      syncTickerRef.current = null;
    }
    setSyncPlaying(false);
  }, [getSyncVideos]);

  const handleSyncRateChange = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) => {
      const nextRate = Number(event.target.value);
      setSyncPlaybackRate(nextRate);
      syncSetRateAll(nextRate);
    },
    [syncSetRateAll],
  );

  const handleMasterEnded = useCallback(() => {
    if (followTraining && nextGroup) {
      setAutoAdvancePending(true);
      setSelectedGroupKey(nextGroup.key);
      setTrainHint(`已切换下一组素材：${nextGroup.label}`);
      return;
    }
    handleSyncPause();
  }, [followTraining, handleSyncPause, nextGroup]);

  useEffect(() => {
    if (!syncPlaying) {
      if (syncTickerRef.current !== null) {
        window.clearInterval(syncTickerRef.current);
        syncTickerRef.current = null;
      }
      return undefined;
    }
    if (syncTickerRef.current !== null) {
      window.clearInterval(syncTickerRef.current);
    }
    syncTickerRef.current = window.setInterval(() => {
      syncFromMaster(false);
    }, 80);
    return () => {
      if (syncTickerRef.current !== null) {
        window.clearInterval(syncTickerRef.current);
        syncTickerRef.current = null;
      }
    };
  }, [syncFromMaster, syncPlaying]);

  useEffect(() => {
    if (!autoAdvancePending || !syncReady) {
      return;
    }
    setAutoAdvancePending(false);
    syncSeekAll(0);
    void handleSyncPlay();
  }, [autoAdvancePending, handleSyncPlay, syncReady, syncSeekAll]);

  useEffect(() => {
    setSyncCurrentTime(0);
    setSyncDuration(0);
    setSyncPlaying(false);
    setPrefetchHint('');
    syncCurrentTimeRef.current = 0;
    syncUiUpdateAtRef.current = 0;
    sourceVideoRefs.current = {};
    pose2dVideoRefs.current = {};
    pose3dVideoRef.current = null;
    if (syncTickerRef.current !== null) {
      window.clearInterval(syncTickerRef.current);
      syncTickerRef.current = null;
    }
  }, [selectedDatasetId, selectedGroupKey]);

  useEffect(() => {
    autoPlayedJobRef.current = '';
    autoPlayStartedRef.current = '';
  }, [selectedDatasetId]);

  useEffect(() => {
    setAutoAdvancePending(false);
  }, [selectedDatasetId]);

  useEffect(() => {
    if (!followTraining || !syncReady || syncPlaying) {
      return;
    }
    const currentKey = `${followTrainJobId || 'running'}:${selectedGroupKey}`;
    if (autoPlayStartedRef.current === currentKey) {
      return;
    }
    autoPlayStartedRef.current = currentKey;
    syncSeekAll(0);
    void handleSyncPlay();
  }, [followTrainJobId, followTraining, handleSyncPlay, selectedGroupKey, syncPlaying, syncReady, syncSeekAll]);

  useEffect(() => {
    if (!syncPlaying || !nextGroup) {
      return;
    }
    const targetSamples = nextGroup.samples.slice(0, 3);
    if (targetSamples.length === 0) {
      return;
    }
    const needsPrefetch = targetSamples.some((sample) => !posePreviewCacheRef.current[sample.path]);
    if (!needsPrefetch) {
      setPrefetchHint('');
      return;
    }
    let cancelled = false;
    setPrefetchHint(`后台预加载下一组：${nextGroup.label}`);
    const run = async () => {
      await ensureGroupPosePreview(targetSamples, {
        showLoading: false,
        updateError: false,
      });
      if (!cancelled) {
        setPrefetchHint(`下一组已就绪：${nextGroup.label}`);
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [ensureGroupPosePreview, nextGroup, syncPlaying]);

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
      setPrefetchHint('');
      return;
    }

    if (currentJob.status === 'succeeded') {
      setFollowProgress(1);
      setFollowCurrentStep((prev) => (followTotalStep > 0 ? followTotalStep : prev));
      setFollowTraining(false);
      setTrainHint(
        syncReady
          ? `训练完成：${followTrainJobId}，开始同步播放当前素材组。`
          : `训练完成：${followTrainJobId}，等待骨架加载完成后可播放。`,
      );
      progressValueRef.current = 1;
      setProgressUpdatedAt(Date.now());
      setProgressWatchTs(Date.now());
      if (syncReady) {
        syncSeekAll(0);
        void handleSyncPlay();
      }
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
    if (!syncReady) {
      return;
    }
    if (autoPlayedJobRef.current === latestTrainJob.job_id) {
      return;
    }
    autoPlayedJobRef.current = latestTrainJob.job_id;
    syncSeekAll(0);
    void handleSyncPlay();
    setTrainHint(`训练完成：${latestTrainJob.job_id}，开始同步播放当前素材组。`);
  }, [followProgress, followTrainJobId, handleSyncPlay, latestTrainJob, syncReady, syncSeekAll]);

  const handleStartTraining = useCallback(async () => {
    if (!selectedDatasetId) {
      return;
    }
    const trainConfigPath = selectedDataset?.train_config?.trim() || 'configs/train.yaml';
      setTrainSubmitting(true);
      setTrainHint('');
      setTrainEvents([]);
      setPrefetchHint('');
    autoPlayedJobRef.current = '';
    try {
      const jobId = await createTrainJob({
        dataset_id: selectedDatasetId,
        config: trainConfigPath,
        export_onnx: true,
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
  }, [refreshCore, selectedDataset?.train_config, selectedDatasetId]);

  const handleRegenerateCurrentGroup = useCallback(async () => {
    if (!selectedDatasetId || currentGroupSamples.length === 0) {
      return;
    }
    setRegeneratingPose(true);
    setPosePreviewError('');
    try {
      for (const sample of currentGroupSamples) {
        delete posePreviewCacheRef.current[sample.path];
        delete posePreviewPendingRef.current[sample.path];
      }
      setPosePreviewMap((prev) => {
        const next = { ...prev };
        for (const sample of currentGroupSamples) {
          delete next[sample.path];
        }
        return next;
      });

      const refreshed = await Promise.all(
        currentGroupSamples.map(async (sample) => {
          try {
            const payload = await fetchPosePreview(selectedDatasetId, sample.path, true);
            return { path: sample.path, payload };
          } catch {
            return { path: sample.path, payload: null };
          }
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
            {prefetchHint && (
              <span className="rounded-md border border-zinc-200 bg-stone-50 px-2 py-1 text-zinc-600">
                {prefetchHint}
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
                四联同步可视化
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
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm" className="h-7 px-2 text-xs" onClick={selectAllGroups}>
                    全选
                  </Button>
                  <span className="rounded-md border border-zinc-200 bg-white px-2 py-1 text-xs text-zinc-600">
                    已选 {selectedGroupKeys.length}/{sourceGroups.length}
                  </span>
                </div>
              </div>
              <div className="max-h-56 space-y-2 overflow-auto rounded-lg border border-zinc-200 bg-white p-2">
                {sourceGroups.length > 0 ? (
                  sourceGroups.map((group) => {
                    const selected = selectedGroupKey === group.key;
                    const checked = selectedGroupKeys.includes(group.key);
                    const isTraining = followTraining && asyncTrainGroupKey === group.key;
                    const statusLabel = isTraining
                      ? '训练中'
                      : group.completedViews >= group.totalViews
                        ? '已完成'
                        : group.completedViews > 0
                          ? '部分完成'
                          : '未生成';
                    const statusClass = selected
                      ? 'border border-white/40 bg-white/10 text-white'
                      : isTraining
                        ? 'border border-sky-200 bg-sky-50 text-sky-700'
                        : group.completedViews >= group.totalViews
                          ? 'border border-emerald-200 bg-emerald-50 text-emerald-700'
                          : group.completedViews > 0
                            ? 'border border-amber-200 bg-amber-50 text-amber-700'
                            : 'border border-zinc-200 bg-white text-zinc-600';
                    return (
                      <div
                        key={group.key}
                        className={`w-full rounded-lg border px-3 py-2 transition ${
                          selected ? 'border-zinc-900 bg-zinc-900 text-white' : 'border-zinc-200 bg-stone-50 text-zinc-700'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => toggleGroupSelected(group.key)}
                            className="h-4 w-4 rounded border-zinc-300 accent-zinc-900"
                          />
                          <button
                            type="button"
                            onClick={() => setSelectedGroupKey(group.key)}
                            className="min-w-0 flex-1 text-left"
                          >
                            <span className="block truncate text-sm font-semibold">{group.label}</span>
                          </button>
                          <span
                            className={`rounded-md px-1.5 py-0.5 text-[11px] font-semibold ${statusClass}`}
                          >
                            {statusLabel}
                          </span>
                          {selected && (
                            <span className="rounded-md border border-white/40 bg-white/10 px-1.5 py-0.5 text-[11px] font-semibold text-white">
                              当前预览
                            </span>
                          )}
                        </div>
                        <div className={`mt-1 text-xs ${selected ? 'text-zinc-200' : 'text-zinc-500'}`}>
                          视角数 {group.samples.length} · 骨架 {group.completedViews}/{group.totalViews} · 体积 {formatBytes(group.totalSizeBytes)}
                        </div>
                      </div>
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
                {posePreviewLoading && (
                  <span className="rounded-md border border-zinc-200 bg-white px-2 py-1 text-xs text-zinc-500">
                    骨架生成中...
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
            </div>

            <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-4 xl:auto-rows-[minmax(236px,auto)]">
              {viewSlots.map((slot, index) => (
                <div key={`source-${index}`} className={`self-start rounded-xl border border-zinc-200 bg-stone-50 p-3 ${SOURCE_COLUMN_CLASSES[index]}`}>
                  <h3 className="mb-2 text-sm font-bold text-zinc-800">
                    素材 {index + 1} · {slot.cameraLabel}
                  </h3>
                  {slot.sourceVideoUrl ? (
                    <video
                      key={`source-video-${slot.sample?.path || index}`}
                      ref={(node) => {
                        if (slot.sample) {
                          sourceVideoRefs.current[slot.sample.path] = node;
                        }
                      }}
                      controls={false}
                      preload="auto"
                      playsInline
                      muted
                      onLoadedMetadata={(event) => handleSyncLoadedMetadata(event.currentTarget)}
                      onTimeUpdate={() => {
                        if (index === 0) {
                          handleSourceTimeUpdate();
                        }
                      }}
                      onPlay={() => {
                        if (index === 0) {
                          setSyncPlaying(true);
                          syncFromMaster(true);
                        }
                      }}
                      onPause={() => {
                        if (index === 0) {
                          handleSyncPause();
                        }
                      }}
                      onEnded={() => {
                        if (index === 0) {
                          handleMasterEnded();
                        }
                      }}
                      className="h-56 w-full rounded-lg border border-zinc-200 bg-zinc-100 object-cover"
                    >
                      <source src={slot.sourceVideoUrl} type="video/mp4" />
                      当前浏览器无法播放视频，请检查编解码格式。
                    </video>
                  ) : (
                    <div className="flex h-56 w-full items-center justify-center rounded-lg border border-dashed border-zinc-300 bg-white text-sm text-zinc-500">
                      {previewLoading ? '正在加载素材...' : '当前分组无该视角素材'}
                    </div>
                  )}
                </div>
              ))}

              {viewSlots.map((slot, index) => (
                <div key={`pose2d-${index}`} className={`self-start rounded-xl border border-zinc-200 bg-stone-50 p-3 ${POSE2D_COLUMN_CLASSES[index]}`}>
                  <h3 className="mb-2 text-sm font-bold text-zinc-800">
                    2D骨架 {index + 1} · {slot.cameraLabel}
                  </h3>
                  {slot.pose2dVideoUrl ? (
                    <video
                      key={`pose2d-video-${slot.sample?.path || index}`}
                      ref={(node) => {
                        if (slot.sample) {
                          pose2dVideoRefs.current[slot.sample.path] = node;
                        }
                      }}
                      controls={false}
                      preload="auto"
                      playsInline
                      muted
                      onLoadedMetadata={(event) => handleSyncLoadedMetadata(event.currentTarget)}
                      className="h-56 w-full rounded-lg border border-zinc-200 bg-zinc-100 object-cover"
                    >
                      <source src={slot.pose2dVideoUrl} type="video/mp4" />
                      当前浏览器无法播放视频，请检查编解码格式。
                    </video>
                  ) : (
                    <div className="flex h-56 w-full items-center justify-center rounded-lg border border-dashed border-zinc-300 bg-white text-sm text-zinc-500">
                      {posePreviewLoading ? '正在生成 2D 骨架...' : '当前视角暂无 2D 骨架'}
                    </div>
                  )}
                </div>
              ))}

              <div className="self-stretch rounded-xl border border-zinc-200 bg-stone-50 p-3 xl:col-start-4 xl:row-start-1 xl:row-span-2">
                <h3 className="mb-2 text-sm font-bold text-zinc-800">
                  3D骨架（融合）
                  {syncPose3dIsFallback && (
                    <span className="ml-2 rounded-md border border-zinc-300 bg-white px-1.5 py-0.5 text-[11px] font-semibold text-zinc-600">
                      回退样例
                    </span>
                  )}
                </h3>
                {syncPose3dVideoUrl ? (
                  <video
                    key={`pose3d-${syncPose3dVideoUrl || 'empty'}`}
                    ref={pose3dVideoRef}
                    controls={false}
                    preload="auto"
                    playsInline
                    muted
                    onLoadedMetadata={(event) => handleSyncLoadedMetadata(event.currentTarget)}
                    className="h-[472px] w-full rounded-lg border border-zinc-200 bg-zinc-100 object-cover"
                  >
                    <source src={syncPose3dVideoUrl} type="video/mp4" />
                    当前浏览器无法播放视频，请检查编解码格式。
                  </video>
                ) : (
                  <div className="flex h-[472px] w-full items-center justify-center rounded-lg border border-dashed border-zinc-300 bg-white text-sm text-zinc-500">
                    {posePreviewLoading ? '正在生成 3D 骨架...' : '当前分组暂无 3D 骨架'}
                  </div>
                )}
              </div>
            </div>

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
