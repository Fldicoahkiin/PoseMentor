import { useEffect, useMemo, useState } from 'react';
import type { DatasetItem, StandardItem } from '../lib/api';
import type { SourceGroup } from './useSourceGroups';

export function useDatasetSelection(
  datasets: DatasetItem[],
  standards: StandardItem[],
  sourceGroups: SourceGroup[],
) {
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [selectedStandardId, setSelectedStandardId] = useState('');
  const [selectedGroupKey, setSelectedGroupKey] = useState('');

  useEffect(() => {
    setSelectedDatasetId((prev) => {
      if (datasets.length === 0) return '';
      const exists = datasets.some((item) => item.id === prev);
      return exists ? prev : datasets[0].id;
    });
  }, [datasets]);

  useEffect(() => {
    setSelectedStandardId((prev) => {
      if (standards.length === 0) return '';
      const exists = standards.some((item) => item.id === prev);
      return exists ? prev : standards[0].id;
    });
  }, [standards]);

  useEffect(() => {
    setSelectedGroupKey((prev) => {
      if (sourceGroups.length === 0) return '';
      const exists = sourceGroups.some((group) => group.key === prev);
      if (exists) return prev;
      const defaultGroup =
        [...sourceGroups].sort((left, right) => right.samples.length - left.samples.length)[0] ?? sourceGroups[0];
      return defaultGroup.key;
    });
  }, [sourceGroups]);

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
    setSelectedDatasetId,
    selectedStandardId,
    setSelectedStandardId,
    selectedGroupKey,
    setSelectedGroupKey,
    selectedDataset,
    selectedStandard,
  };
}
