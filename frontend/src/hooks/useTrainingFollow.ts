import { useEffect, useMemo, useRef, useState } from 'react';
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
  const [progressUpdatedAt, setProgressUpdatedAt] = useState(0);
  const [progressWatchTs, setProgressWatchTs] = useState(0);
  const [pendingAutoPlayJobId, setPendingAutoPlayJobId] = useState('');
  const progressValueRef = useRef(0);

  // 自动跟踪运行中的训练任务
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
    }
  }, [followTraining, latestTrainJob, progressUpdatedAt]);

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

  // 检测训练完成/失败
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
    }
  }, [followTotalStep, followTrainJobId, followTraining, jobs, syncReady]);

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
  };
}
