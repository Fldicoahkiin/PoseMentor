/**
 * 视频同步播放的公共辅助函数和常量。
 * DemoPage 和 hooks 共享此模块，避免重复定义。
 */

export const CAMERA_TOKEN_PATTERN = /_c(\d+)_/i;
export const SYNC_DRIFT_TOLERANCE = 0.05;
export const SYNC_TICK_MS = 40;
export const SYNC_PAUSE_SETTLE_MS = 72;
export const TRAIN_PROGRESS_STALL_MS = 20_000;

export function seekVideo(video: HTMLVideoElement, timeSeconds: number): void {
  try {
    if (typeof video.fastSeek === 'function') {
      video.fastSeek(timeSeconds);
      return;
    }
  } catch {
    // fastSeek 失败时回退为 currentTime 赋值
  }
  video.currentTime = timeSeconds;
}

export function pickMedian(values: number[], fallback = 0): number {
  const ordered = values.filter((value) => Number.isFinite(value)).sort((left, right) => left - right);
  if (ordered.length === 0) {
    return fallback;
  }
  const middle = Math.floor(ordered.length / 2);
  if (ordered.length % 2 === 1) {
    return ordered[middle] ?? fallback;
  }
  const left = ordered[middle - 1] ?? fallback;
  const right = ordered[middle] ?? fallback;
  return (left + right) / 2;
}

export function waitForVideoPlayable(video: HTMLVideoElement, timeoutMs = 4000): Promise<void> {
  if (video.readyState >= 2) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    let done = false;
    let timerId = 0;
    const cleanup = () => {
      video.removeEventListener('loadeddata', finish);
      video.removeEventListener('canplay', finish);
      video.removeEventListener('error', finish);
      window.clearTimeout(timerId);
    };
    const finish = () => {
      if (done) return;
      done = true;
      cleanup();
      resolve();
    };
    timerId = window.setTimeout(finish, timeoutMs);
    video.addEventListener('loadeddata', finish, { once: true });
    video.addEventListener('canplay', finish, { once: true });
    video.addEventListener('error', finish, { once: true });
  });
}

export function normalizeSequenceKey(pathValue: string): string {
  const name = pathValue.split('/').at(-1) ?? pathValue;
  return name.replace(/\.mp4$/i, '').replace(CAMERA_TOKEN_PATTERN, '_cAll_');
}

export function formatBytes(sizeBytes: number): string {
  if (sizeBytes < 1024) {
    return `${sizeBytes} B`;
  }
  if (sizeBytes < 1024 * 1024) {
    return `${(sizeBytes / 1024).toFixed(1)} KB`;
  }
  if (sizeBytes < 1024 * 1024 * 1024) {
    return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(sizeBytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function formatTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

export function formatClock(totalSeconds: number): string {
  const value = Number.isFinite(totalSeconds) ? Math.max(0, totalSeconds) : 0;
  const seconds = Math.floor(value % 60);
  const minutes = Math.floor((value / 60) % 60);
  const hours = Math.floor(value / 3600);
  if (hours > 0) {
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds
      .toString()
      .padStart(2, '0')}`;
  }
  return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

export function formatDecimal(value: number | undefined | null, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '-';
  }
  return Number(value).toFixed(digits);
}

export function formatFrameOffset(value: number | undefined | null): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '-';
  }
  const numberValue = Math.trunc(value);
  return `${numberValue > 0 ? '+' : ''}${numberValue}f`;
}
