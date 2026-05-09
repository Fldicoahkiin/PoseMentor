import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { fetchJobProgress, type JobItem } from '../lib/api';
import { TRAIN_PROGRESS_STALL_MS } from '../lib/videoUtils';

export type TrainingFollowState = {
  followTraining: boolean;
  setFollowTraining: (v: boolean) => void;
  followTrainJobId: string;
  followProgress: number;
  followCurrentStep: number;
  followTotalStep: number;
  trainHint: string;
  setTrainHint: (v: string) => void;
  trainEvents: string[];
  trainingStalled: boolean;
  progressPercent: number;
  progressTextPercent: number;
  followStepLabel: string;
  pendingAutoPlayJobId: string;
  setPendingAutoPlayJobId: (v: string) => void;
  startFollowing: (jobId: string) => void;
};

export function useTrainingFollow(
  jobs: JobItem[],
  latestTrainJob: JobItem | null,
  syncReady: boolean,
): TrainingFollowState {
  const [followTraining, setFollowTraining] = useState(false);
  const [followTrainJobId, setFollowTrainJobId] = useState('');
  const [followProgress, setFollowProgress] = useState(0);
  const [followCurrentStep, setFollowCurrentStep] = useState(0);
  const [followTotalStep, setFollowTotalStep] = useState(0);
  const [trainEvents, setTrainEvents] = useState<string[]>([]);
  const [trainHint, setTrainHint] = useState('');
  const [trainingStalled, setTrainingStalled] = useState(false);
  const [pendingAutoPlayJobId, setPendingAutoPlayJobId] = useState('');

  // 时间戳只用于 stall 检测，不参与渲染，用 refs 避免 purity 问题
  const progressValueRef = useRef(0);
  const progressUpdatedAtRef = useRef(0);
  const progressWatchTsRef = useRef(0);

  // 自动跟踪运行中的训练任务（渲染期间纯 setState）
  const [prevLatestJobId, setPrevLatestJobId] = useState<string | null>(null);
  const currentLatestJobId = latestTrainJob?.job_id ?? null;
  if (currentLatestJobId !== prevLatestJobId) {
    setPrevLatestJobId(currentLatestJobId);
    if (followTraining) {
      if (latestTrainJob) {
        setFollowTrainJobId(latestTrainJob.job_id);
      }
    } else if (latestTrainJob?.status === 'running') {
      setFollowTraining(true);
      setFollowTrainJobId(latestTrainJob.job_id);
    }
  }

  // 新检测到运行中任务时初始化时间戳（effect，允许 refs + Date.now）
  useEffect(() => {
    if (followTraining && progressUpdatedAtRef.current === 0) {
      const now = Date.now();
      progressUpdatedAtRef.current = now;
      progressWatchTsRef.current = now;
    }
  }, [followTraining]);

  // 轮询训练进度
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
        progressWatchTsRef.current = now;
        if (progressValue >= progressValueRef.current + 0.001) {
          progressValueRef.current = progressValue;
          progressUpdatedAtRef.current = now;
        }
        // stall 检测：在轮询回调中计算，避免渲染期间读 refs
        const stalled = progressValue < 0.999
          && progressUpdatedAtRef.current > 0
          && now - progressUpdatedAtRef.current > TRAIN_PROGRESS_STALL_MS;
        setTrainingStalled(stalled);
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

  // 检测训练完成/失败（渲染期间纯 setState）
  const currentFollowedJob = followTrainJobId
    ? jobs.find((item) => item.job_id === followTrainJobId)
    : null;
  const followedJobStatus = currentFollowedJob?.status;
  const [prevFollowedJobStatus, setPrevFollowedJobStatus] = useState<string | undefined>(undefined);
  if (followedJobStatus !== prevFollowedJobStatus) {
    setPrevFollowedJobStatus(followedJobStatus);
    if (followTraining && followedJobStatus === 'failed') {
      setFollowTraining(false);
      setTrainHint(`训练失败：${followTrainJobId}`);
      setTrainEvents([]);
      setTrainingStalled(false);
    } else if (followTraining && followedJobStatus === 'succeeded') {
      setFollowProgress(1);
      setFollowCurrentStep(followTotalStep > 0 ? followTotalStep : followCurrentStep);
      setFollowTraining(false);
      setTrainHint(
        syncReady
          ? `训练完成：${followTrainJobId}，正在准备同步播放。`
          : `训练完成：${followTrainJobId}，等待骨架加载完成后可播放。`,
      );
      setTrainingStalled(false);
      setPendingAutoPlayJobId(followTrainJobId);
    }
  }

  // 训练完成时更新 refs（effect，允许 refs + Date.now）
  useEffect(() => {
    if (followedJobStatus === 'succeeded') {
      progressValueRef.current = 1;
      const now = Date.now();
      progressUpdatedAtRef.current = now;
      progressWatchTsRef.current = now;
    }
  }, [followedJobStatus]);

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

  const startFollowing = useCallback((jobId: string) => {
    setFollowTraining(true);
    setFollowTrainJobId(jobId);
    setFollowProgress(0);
    setFollowCurrentStep(0);
    setFollowTotalStep(0);
    setTrainEvents([]);
    setTrainingStalled(false);
    progressValueRef.current = 0;
    const now = Date.now();
    progressUpdatedAtRef.current = now;
    progressWatchTsRef.current = now;
  }, []);

  return {
    followTraining,
    setFollowTraining,
    followTrainJobId,
    followProgress,
    followCurrentStep,
    followTotalStep,
    trainHint,
    setTrainHint,
    trainEvents,
    trainingStalled,
    progressPercent,
    progressTextPercent,
    followStepLabel,
    pendingAutoPlayJobId,
    setPendingAutoPlayJobId,
    startFollowing,
  };
}
