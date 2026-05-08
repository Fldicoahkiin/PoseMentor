import { useCallback, useEffect, useRef, useState, type ChangeEvent } from 'react';
import {
  SYNC_DRIFT_TOLERANCE,
  SYNC_TICK_MS,
  SYNC_PAUSE_SETTLE_MS,
  seekVideo,
  pickMedian,
  waitForVideoPlayable,
} from '../lib/videoUtils';

export type SyncPlaybackControls = {
  syncPlaying: boolean;
  syncCurrentTime: number;
  syncDuration: number;
  syncPlaybackRate: number;
  syncReady: boolean;
  sourceVideoRefs: React.MutableRefObject<Record<string, HTMLVideoElement | null>>;
  syncCurrentTimeRef: React.MutableRefObject<number>;
  syncPlayingRef: React.MutableRefObject<boolean>;
  setSyncPlaybackRate: (rate: number) => void;
  setSyncPlaying: (v: boolean) => void;
  setSyncCurrentTime: (v: number) => void;
  setSyncDuration: (v: number) => void;
  getMasterSourceVideo: () => HTMLVideoElement | null;
  getSyncVideos: () => HTMLVideoElement[];
  getSyncTimes: () => number[];
  getFollowerVideos: () => HTMLVideoElement[];
  syncSeekAll: (time: number) => void;
  syncSetRateAll: (rate: number) => void;
  syncFromMaster: (force: boolean) => void;
  handleSyncPlay: () => Promise<boolean>;
  handleSyncPause: () => void;
  handleSyncRateChange: (event: ChangeEvent<HTMLSelectElement>) => void;
  handleSyncLoadedMetadata: (video: HTMLVideoElement) => void;
  handleVideoLoadedData: (video: HTMLVideoElement) => void;
  handleSourceTimeUpdate: () => void;
  clearSyncTicker: () => void;
  clearSyncPauseSettleTimer: () => void;
  resetSyncState: () => void;
};

export function useSyncPlayback(currentGroupSamplePaths: string[]): SyncPlaybackControls {
  const [syncPlaying, setSyncPlaying] = useState(false);
  const [syncCurrentTime, setSyncCurrentTime] = useState(0);
  const [syncDuration, setSyncDuration] = useState(0);
  const [syncPlaybackRate, setSyncPlaybackRate] = useState(1);

  const sourceVideoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const syncTickerRef = useRef<number | null>(null);
  const syncPauseSettleRef = useRef<number | null>(null);
  const syncCurrentTimeRef = useRef(0);
  const syncUiUpdateAtRef = useRef(0);
  const syncPlayingRef = useRef(false);
  const syncPauseGuardRef = useRef(false);

  const syncReady = syncDuration > 0;

  const clearSyncTicker = useCallback(() => {
    if (syncTickerRef.current !== null) {
      window.clearInterval(syncTickerRef.current);
      syncTickerRef.current = null;
    }
  }, []);

  const clearSyncPauseSettleTimer = useCallback(() => {
    if (syncPauseSettleRef.current !== null) {
      window.clearTimeout(syncPauseSettleRef.current);
      syncPauseSettleRef.current = null;
    }
    syncPauseGuardRef.current = false;
  }, []);

  const getMasterSourceVideo = useCallback((): HTMLVideoElement | null => {
    if (currentGroupSamplePaths.length === 0) return null;
    return sourceVideoRefs.current[currentGroupSamplePaths[0]] ?? null;
  }, [currentGroupSamplePaths]);

  const getSyncVideos = useCallback((): HTMLVideoElement[] => {
    return currentGroupSamplePaths
      .map((path) => sourceVideoRefs.current[path])
      .filter((v): v is HTMLVideoElement => v !== null && v !== undefined);
  }, [currentGroupSamplePaths]);

  const getFollowerVideos = useCallback((): HTMLVideoElement[] => {
    return currentGroupSamplePaths
      .slice(1)
      .map((path) => sourceVideoRefs.current[path])
      .filter((v): v is HTMLVideoElement => v !== null && v !== undefined);
  }, [currentGroupSamplePaths]);

  const getSyncTimes = useCallback((): number[] => {
    return getSyncVideos()
      .filter((v) => v.readyState >= 2)
      .map((v) => v.currentTime);
  }, [getSyncVideos]);

  const updateSyncCurrentTime = useCallback((time: number, force: boolean) => {
    syncCurrentTimeRef.current = time;
    const now = performance.now();
    if (force || now - syncUiUpdateAtRef.current > 80) {
      syncUiUpdateAtRef.current = now;
      setSyncCurrentTime(time);
    }
  }, []);

  const recomputeSyncDuration = useCallback(() => {
    const durations = getSyncVideos()
      .filter((v) => v.readyState >= 1 && Number.isFinite(v.duration) && v.duration > 0.05)
      .map((v) => v.duration);
    if (durations.length === 0) {
      setSyncDuration(0);
      return;
    }
    setSyncDuration(Math.min(...durations));
  }, [getSyncVideos]);

  const syncSeekAll = useCallback(
    (time: number) => {
      getSyncVideos().forEach((element) => seekVideo(element, time));
      updateSyncCurrentTime(time, true);
    },
    [getSyncVideos, updateSyncCurrentTime],
  );

  const syncSetRateAll = useCallback(
    (rate: number) => {
      const master = getMasterSourceVideo();
      if (master) master.playbackRate = rate;
      getFollowerVideos().forEach((element) => {
        element.playbackRate = rate;
      });
    },
    [getFollowerVideos, getMasterSourceVideo],
  );

  const syncFromMaster = useCallback(
    (force: boolean) => {
      const master = getMasterSourceVideo();
      if (!master || syncPauseGuardRef.current) return;
      const sourceTime = master.currentTime;
      const tolerance = force ? 0.008 : SYNC_DRIFT_TOLERANCE;
      getFollowerVideos().forEach((element) => {
        if (element.readyState < 2) return;
        if (Math.abs(element.currentTime - sourceTime) > tolerance) {
          seekVideo(element, sourceTime);
        }
        if (Math.abs(element.playbackRate - syncPlaybackRate) > 0.001) {
          element.playbackRate = syncPlaybackRate;
        }
        if (!master.paused && element.paused) {
          void element.play().catch(() => undefined);
        }
      });
      if (Math.abs(master.playbackRate - syncPlaybackRate) > 0.001) {
        master.playbackRate = syncPlaybackRate;
      }
      updateSyncCurrentTime(sourceTime, force);
    },
    [getFollowerVideos, getMasterSourceVideo, syncPlaybackRate, updateSyncCurrentTime],
  );

  const handleSyncPlay = useCallback(async (): Promise<boolean> => {
    const master = getMasterSourceVideo();
    if (!master) return false;
    const videos = getSyncVideos();
    if (videos.length === 0) return false;

    clearSyncTicker();
    clearSyncPauseSettleTimer();
    syncPauseGuardRef.current = false;

    const anchorTime = pickMedian(getSyncTimes(), syncCurrentTimeRef.current);
    await Promise.all(videos.map((element) => waitForVideoPlayable(element)));
    syncSeekAll(anchorTime);
    syncSetRateAll(syncPlaybackRate);

    await Promise.allSettled(
      videos.map(async (element) => {
        if (Math.abs(element.currentTime - anchorTime) > 0.008) {
          seekVideo(element, anchorTime);
        }
        if (Math.abs(element.playbackRate - syncPlaybackRate) > 0.001) {
          element.playbackRate = syncPlaybackRate;
        }
        if (element.paused) {
          await element.play();
        }
      }),
    );

    if (master.paused) {
      syncPlayingRef.current = false;
      setSyncPlaying(false);
      return false;
    }
    syncPlayingRef.current = true;
    setSyncPlaying(true);

    syncFromMaster(true);
    return true;
  }, [clearSyncPauseSettleTimer, clearSyncTicker, getMasterSourceVideo, getSyncTimes, getSyncVideos, syncFromMaster, syncPlaybackRate, syncSeekAll, syncSetRateAll]);

  const handleSyncPause = useCallback(() => {
    if (syncPauseGuardRef.current) return;
    syncPauseGuardRef.current = true;
    syncPlayingRef.current = false;
    clearSyncTicker();
    clearSyncPauseSettleTimer();

    const pauseTime = pickMedian(getSyncTimes(), syncCurrentTimeRef.current);
    const videos = getSyncVideos();
    videos.forEach((element) => {
      element.pause();
      if (Math.abs(element.playbackRate - syncPlaybackRate) > 0.001) {
        element.playbackRate = syncPlaybackRate;
      }
    });
    syncSeekAll(pauseTime);
    syncPauseSettleRef.current = window.setTimeout(() => {
      syncSeekAll(pauseTime);
      syncPauseGuardRef.current = false;
      syncPauseSettleRef.current = null;
    }, SYNC_PAUSE_SETTLE_MS);
    setSyncPlaying(false);
  }, [clearSyncPauseSettleTimer, clearSyncTicker, getSyncTimes, getSyncVideos, syncPlaybackRate, syncSeekAll]);

  const handleSyncRateChange = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) => {
      const nextRate = Number(event.target.value);
      setSyncPlaybackRate(nextRate);
      syncSetRateAll(nextRate);
    },
    [syncSetRateAll],
  );

  const handleSyncLoadedMetadata = useCallback(
    (video: HTMLVideoElement) => {
      video.playbackRate = syncPlaybackRate;
      const anchorTime = syncCurrentTimeRef.current;
      if (anchorTime > 0.01 && Math.abs(video.currentTime - anchorTime) > 0.02) {
        seekVideo(video, anchorTime);
      }
      if (video !== getMasterSourceVideo() && !syncPlayingRef.current && !video.paused) {
        video.pause();
      }
      recomputeSyncDuration();
    },
    [getMasterSourceVideo, recomputeSyncDuration, syncPlaybackRate],
  );

  const handleVideoLoadedData = useCallback((video: HTMLVideoElement) => {
    if (syncCurrentTimeRef.current > 0.01 || video.readyState < 2 || video.duration <= 0.05) {
      return;
    }
    if (video.currentTime > 0.001) return;
    try {
      video.currentTime = 0.001;
    } catch {
      return;
    }
  }, []);

  const handleSourceTimeUpdate = useCallback(() => {
    const source = getMasterSourceVideo();
    if (!source || syncPauseGuardRef.current) return;
    syncFromMaster(false);
  }, [getMasterSourceVideo, syncFromMaster]);

  // sync ticker effect
  useEffect(() => {
    syncPlayingRef.current = syncPlaying;
    if (!syncPlaying) {
      clearSyncTicker();
      return undefined;
    }
    clearSyncTicker();
    syncTickerRef.current = window.setInterval(() => {
      syncFromMaster(false);
    }, SYNC_TICK_MS);
    return () => {
      clearSyncTicker();
    };
  }, [clearSyncTicker, syncFromMaster, syncPlaying]);

  const resetSyncState = useCallback(() => {
    setSyncCurrentTime(0);
    setSyncDuration(0);
    setSyncPlaying(false);
    syncCurrentTimeRef.current = 0;
    syncUiUpdateAtRef.current = 0;
    syncPlayingRef.current = false;
    syncPauseGuardRef.current = false;
    sourceVideoRefs.current = {};
    clearSyncTicker();
    clearSyncPauseSettleTimer();
  }, [clearSyncPauseSettleTimer, clearSyncTicker]);

  return {
    syncPlaying,
    syncCurrentTime,
    syncDuration,
    syncPlaybackRate,
    syncReady,
    sourceVideoRefs,
    syncCurrentTimeRef,
    syncPlayingRef,
    setSyncPlaybackRate,
    setSyncPlaying,
    setSyncCurrentTime,
    setSyncDuration,
    getMasterSourceVideo,
    getSyncVideos,
    getSyncTimes,
    getFollowerVideos,
    syncSeekAll,
    syncSetRateAll,
    syncFromMaster,
    handleSyncPlay,
    handleSyncPause,
    handleSyncRateChange,
    handleSyncLoadedMetadata,
    handleVideoLoadedData,
    handleSourceTimeUpdate,
    clearSyncTicker,
    clearSyncPauseSettleTimer,
    resetSyncState,
  };
}
