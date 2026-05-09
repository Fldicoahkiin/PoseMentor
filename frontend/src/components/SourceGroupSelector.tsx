import { formatBytes } from '../lib/videoUtils';
import type { SourceGroup } from '../hooks/useSourceGroups';

interface SourceGroupSelectorProps {
  sourceGroups: SourceGroup[];
  selectedGroupKey: string;
  currentGroupLabel: string | undefined;
  followTraining: boolean;
  asyncTrainGroupKey: string;
  onSelectGroup: (key: string) => void;
}

export function SourceGroupSelector({
  sourceGroups,
  selectedGroupKey,
  currentGroupLabel,
  followTraining,
  asyncTrainGroupKey,
  onSelectGroup,
}: SourceGroupSelectorProps) {
  return (
    <div className="mb-4 rounded-xl border border-zinc-200 bg-stone-50 p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-bold uppercase tracking-wider text-zinc-500">素材组（同一动作不同 camera）</p>
        <span className="rounded-md border border-zinc-200 bg-white px-2 py-1 text-xs text-zinc-600">
          {currentGroupLabel ? `当前：${currentGroupLabel}` : `共 ${sourceGroups.length} 组`}
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
                onClick={() => onSelectGroup(group.key)}
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
  );
}
