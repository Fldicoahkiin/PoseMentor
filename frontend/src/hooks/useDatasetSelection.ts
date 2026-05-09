import { useMemo, useState } from 'react';
import type { DatasetItem, StandardItem } from '../lib/api';
import type { SourceGroup } from './useSourceGroups';

export function useDatasetSelection(
  datasets: DatasetItem[],
  standards: StandardItem[],
  sourceGroups: SourceGroup[],
) {
  const [rawDatasetId, setRawDatasetId] = useState('');
  const [rawStandardId, setRawStandardId] = useState('');
  const [rawGroupKey, setRawGroupKey] = useState('');

  // 派生有效选中值：当列表变化导致选中项不存在时自动回退到首项
  const selectedDatasetId = useMemo(() => {
    if (datasets.length === 0) return '';
    return datasets.some((item) => item.id === rawDatasetId) ? rawDatasetId : datasets[0].id;
  }, [datasets, rawDatasetId]);

  const selectedStandardId = useMemo(() => {
    if (standards.length === 0) return '';
    return standards.some((item) => item.id === rawStandardId) ? rawStandardId : standards[0].id;
  }, [standards, rawStandardId]);

  const selectedGroupKey = useMemo(() => {
    if (sourceGroups.length === 0) return '';
    if (sourceGroups.some((group) => group.key === rawGroupKey)) return rawGroupKey;
    const defaultGroup =
      [...sourceGroups].sort((left, right) => right.samples.length - left.samples.length)[0] ?? sourceGroups[0];
    return defaultGroup.key;
  }, [sourceGroups, rawGroupKey]);

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId) ?? null,
    [datasets, selectedDatasetId],
  );

  const selectedStandard = useMemo(
    () => standards.find((item) => item.id === selectedStandardId) ?? null,
    [selectedStandardId, standards],
  );

  return {
    selectedDatasetId,
    setSelectedDatasetId: setRawDatasetId,
    selectedStandardId,
    setSelectedStandardId: setRawStandardId,
    selectedGroupKey,
    setSelectedGroupKey: setRawGroupKey,
    selectedDataset,
    selectedStandard,
  };
}
