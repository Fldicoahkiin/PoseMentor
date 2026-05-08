import { useMemo } from 'react';
import type { SourcePreviewItem, SourcePreviewPayload } from '../lib/api';

const CAMERA_TOKEN_PATTERN = /_c(\d+)_/i;

function normalizeSequenceKey(pathValue: string): string {
  const name = pathValue.split('/').at(-1) ?? pathValue;
  return name.replace(/\.mp4$/i, '').replace(CAMERA_TOKEN_PATTERN, '_cAll_');
}

export type SourceGroup = {
  key: string;
  samples: SourcePreviewItem[];
  label: string;
  totalSizeBytes: number;
  generatedViews: number;
  completedViews: number;
  totalViews: number;
};

export function useSourceGroups(sourcePreview: SourcePreviewPayload | null): SourceGroup[] {
  return useMemo<SourceGroup[]>(() => {
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
}
