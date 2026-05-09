interface TrainingProgressBarProps {
  followTraining: boolean;
  progressPercent: number;
  progressTextPercent: number;
  followStepLabel: string;
  trainingStalled: boolean;
  followProgress: number;
  trainEvents: string[];
}

export function TrainingProgressBar({
  followTraining,
  progressPercent,
  progressTextPercent,
  followStepLabel,
  trainingStalled,
  followProgress,
  trainEvents,
}: TrainingProgressBarProps) {
  if (!followTraining && followProgress <= 0) {
    return null;
  }

  return (
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
  );
}
