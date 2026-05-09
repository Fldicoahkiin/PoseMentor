import { Pause, Play } from 'lucide-react';
import { Button } from './ui/Button';
import { formatClock } from '../lib/videoUtils';
import type { ChangeEvent } from 'react';

interface PlaybackControlBarProps {
  syncReady: boolean;
  followTraining: boolean;
  syncPlaying: boolean;
  syncCurrentTime: number;
  syncDuration: number;
  syncPlaybackRate: number;
  posePreviewLoading: boolean;
  posePreviewError: string;
  progressTextPercent: number;
  followStepLabel: string;
  groupPrepareDone: number;
  groupPrepareTotal: number;
  currentGroupSampleCount: number;
  videoRoot: string | undefined;
  activeSeqText: string;
  onPlay: () => void;
  onPause: () => void;
  onRateChange: (event: ChangeEvent<HTMLSelectElement>) => void;
  onSeek: (time: number) => void;
}

export function PlaybackControlBar({
  syncReady,
  followTraining,
  syncPlaying,
  syncCurrentTime,
  syncDuration,
  syncPlaybackRate,
  posePreviewLoading,
  posePreviewError,
  progressTextPercent,
  followStepLabel,
  groupPrepareDone,
  groupPrepareTotal,
  currentGroupSampleCount,
  videoRoot,
  activeSeqText,
  onPlay,
  onPause,
  onRateChange,
  onSeek,
}: PlaybackControlBarProps) {
  const disabled = !syncReady || followTraining;

  return (
    <>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <span className="rounded-md border border-zinc-200 bg-stone-50 px-2 py-1">
            视频根目录：{videoRoot || '未检测到'}
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
          <Button
            size="sm"
            onClick={onPlay}
            disabled={disabled}
            className="h-9 gap-2 px-3"
          >
            <Play className="h-4 w-4" aria-hidden="true" />
            <span>播放</span>
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={onPause}
            disabled={disabled}
            className="h-9 gap-2 px-3"
          >
            <Pause className="h-4 w-4" aria-hidden="true" />
            <span>暂停</span>
          </Button>
          <select
            value={syncPlaybackRate}
            onChange={onRateChange}
            disabled={disabled}
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
          {followTraining && (
            <span className="rounded-md border border-sky-200 bg-sky-50 px-2 py-1 text-xs text-sky-700">
              训练进行中，播放锁定
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
        <div className="mt-2 flex items-center gap-2 text-xs text-zinc-600">
          <span className="rounded-md border border-zinc-200 bg-white px-2 py-1">
            当前组预览就绪 {groupPrepareDone}/{groupPrepareTotal || currentGroupSampleCount}
          </span>
          <div className="h-1.5 flex-1 rounded-full bg-zinc-200">
            <div
              className="h-1.5 rounded-full bg-zinc-900 transition-all"
              style={{
                width: `${
                  (groupPrepareTotal > 0 ? (groupPrepareDone / groupPrepareTotal) * 100 : 0).toFixed(2)
                }%`,
              }}
            />
          </div>
        </div>
        <input
          type="range"
          min={0}
          max={syncDuration > 0 ? syncDuration : 1}
          step={0.01}
          value={Math.min(syncCurrentTime, syncDuration > 0 ? syncDuration : syncCurrentTime)}
          onChange={(event) => onSeek(Number(event.target.value))}
          disabled={disabled}
          className="mt-3 h-2 w-full accent-zinc-900"
        />
      </div>
    </>
  );
}
