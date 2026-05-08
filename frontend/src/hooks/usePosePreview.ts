import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchPosePreview, type PosePreviewPayload, type SourcePreviewItem } from '../lib/api';

export type PosePreviewState = {
  posePreviewMap: Record<string, PosePreviewPayload>;
  posePreviewLoading: boolean;
  setPosePreviewLoading: (v: boolean) => void;
  posePreviewError: string;
  setPosePreviewError: (v: string) => void;
  groupPrepareDone: number;
  setGroupPrepareDone: (v: number) => void;
  groupPrepareTotal: number;
  setGroupPrepareTotal: (v: number) => void;
  fetchPosePreviewForSample: (sample: SourcePreviewItem) => Promise<PosePreviewPayload | null>;
  ensureGroupPosePreview: (
    samples: SourcePreviewItem[],
    options?: {
      showLoading?: boolean;
      updateError?: boolean;
      onProgress?: (done: number, total: number) => void;
    },
  ) => Promise<{ missing: string[] }>;
};

export function usePosePreview(
  selectedDatasetId: string,
  markSourcePreviewGenerated: (samplePath: string) => void,
): PosePreviewState {
  const [posePreviewMap, setPosePreviewMap] = useState<Record<string, PosePreviewPayload>>({});
  const [posePreviewLoading, setPosePreviewLoading] = useState(false);
  const [posePreviewError, setPosePreviewError] = useState('');
  const [groupPrepareDone, setGroupPrepareDone] = useState(0);
  const [groupPrepareTotal, setGroupPrepareTotal] = useState(0);

  const posePreviewCacheRef = useRef<Record<string, PosePreviewPayload>>({});
  const posePreviewPendingRef = useRef<Record<string, Promise<PosePreviewPayload | null>>>({});
  const previewDatasetRef = useRef('');

  // 同步 dataset ref
  useEffect(() => {
    previewDatasetRef.current = selectedDatasetId;
  }, [selectedDatasetId]);

  // 同步 cache ref
  useEffect(() => {
    posePreviewCacheRef.current = posePreviewMap;
  }, [posePreviewMap]);

  // dataset 切换时清空缓存
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

  return {
    posePreviewMap,
    posePreviewLoading,
    setPosePreviewLoading,
    posePreviewError,
    setPosePreviewError,
    groupPrepareDone,
    setGroupPrepareDone,
    groupPrepareTotal,
    setGroupPrepareTotal,
    fetchPosePreviewForSample,
    ensureGroupPosePreview,
  };
}

/**
 * 当 currentGroupSamples 变化时自动加载 pose preview。
 * 在 DemoPage 中调用此 effect。
 */
export function useAutoLoadGroupPreview(
  selectedDatasetId: string,
  currentGroupSamples: SourcePreviewItem[],
  ensureGroupPosePreview: PosePreviewState['ensureGroupPosePreview'],
  setGroupPrepareDone: (v: number) => void,
  setGroupPrepareTotal: (v: number) => void,
  setPosePreviewLoading: (v: boolean) => void,
  setPosePreviewError: (v: string) => void,
) {
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
  }, [currentGroupSamples, ensureGroupPosePreview, selectedDatasetId, setGroupPrepareDone, setGroupPrepareTotal, setPosePreviewLoading, setPosePreviewError]);
}
