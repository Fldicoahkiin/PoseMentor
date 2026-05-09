import { RefreshCw } from 'lucide-react';
import { Button } from './ui/Button';
import type { DatasetItem, StandardItem } from '../lib/api';

type StepStatus = 'ready' | 'running' | 'waiting' | 'error';

interface PipelineStep {
  name: string;
  status: StepStatus;
  detail: string;
}

interface WorkbenchHeaderProps {
  health: string;
  loading: boolean;
  previewLoading: boolean;
  failedJobs: number;
  runningJobs: number;
  datasets: DatasetItem[];
  standards: StandardItem[];
  selectedDatasetId: string;
  selectedStandardId: string;
  selectedDatasetMode: string | undefined;
  selectedStandardName: string | undefined;
  videoRoot: string | undefined;
  pipelineSteps: PipelineStep[];
  onRefreshCore: () => void;
  onRefreshPreview: () => void;
  onDatasetChange: (id: string) => void;
  onStandardChange: (id: string) => void;
}

const STEP_STATUS_COLOR: Record<StepStatus, string> = {
  ready: 'bg-emerald-500',
  running: 'bg-amber-500 animate-pulse',
  error: 'bg-rose-500',
  waiting: 'bg-zinc-300',
};

export function WorkbenchHeader({
  health,
  loading,
  previewLoading,
  failedJobs,
  runningJobs,
  datasets,
  standards,
  selectedDatasetId,
  selectedStandardId,
  selectedDatasetMode,
  selectedStandardName,
  videoRoot,
  pipelineSteps,
  onRefreshCore,
  onRefreshPreview,
  onDatasetChange,
  onStandardChange,
}: WorkbenchHeaderProps) {
  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-bold text-zinc-900">训练工作台</h1>
          <span className="flex items-center gap-1.5 text-xs text-zinc-500">
            <span className={`inline-block h-2 w-2 rounded-full ${health === 'ok' ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
            {health === 'ok' ? '在线' : '离线'}
          </span>
          {failedJobs > 0 && <span className="text-xs text-rose-600">失败 {failedJobs}</span>}
          {runningJobs > 0 && <span className="text-xs text-amber-600">运行中 {runningJobs}</span>}
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={onRefreshCore} disabled={loading} className="gap-1.5">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            刷新
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="gap-1.5"
            disabled={previewLoading || !selectedDatasetId}
            onClick={onRefreshPreview}
          >
            <RefreshCw size={14} className={previewLoading ? 'animate-spin' : ''} />
            素材
          </Button>
        </div>
      </div>
      <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
        <div>
          <label htmlFor="selected-dataset" className="mb-1 block text-xs font-bold uppercase tracking-wider text-zinc-500">
            训练数据集
          </label>
          <select
            id="selected-dataset"
            value={selectedDatasetId}
            onChange={(event) => onDatasetChange(event.target.value)}
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
            onChange={(event) => onStandardChange(event.target.value)}
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
        <span>模式 <b className="text-zinc-800">{selectedDatasetMode || '-'}</b></span>
        <span>标准 <b className="text-zinc-800">{selectedStandardName || '-'}</b></span>
        <span>路径 <b className="text-zinc-800">{videoRoot || '-'}</b></span>
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1">
        {pipelineSteps.map((step) => (
          <span
            key={step.name}
            className="flex items-center gap-1 text-[11px] text-zinc-600"
            title={step.detail}
          >
            <span className={`inline-block h-1.5 w-1.5 rounded-full ${STEP_STATUS_COLOR[step.status]}`} />
            {step.name}
          </span>
        ))}
      </div>
    </div>
  );
}
