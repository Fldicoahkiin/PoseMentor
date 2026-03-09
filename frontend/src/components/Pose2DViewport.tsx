import { useEffect, useMemo, useRef, useState } from 'react';

type Pose2DData = {
  fps: number;
  frame_count: number;
  joint_count: number;
  frame_width: number;
  frame_height: number;
  edges: Array<[number, number]>;
  keypoints2d: number[][][];
};

type Pose2DViewportProps = {
  dataUrl: string;
  currentTime: number;
  className?: string;
  emptyText?: string;
  videoElement?: HTMLVideoElement | null;
  playing?: boolean;
};

const EMPTY_PAD_X = 18;
const EMPTY_PAD_Y = 16;
const VIDEO_INSET_X = 0;
const VIDEO_INSET_Y = 0;
const JOINT_RADIUS_MIN = 1.8;
const JOINT_RADIUS_MAX = 3.0;
const JOINT_RADIUS_RATIO = 0.0085;
const EDGE_WIDTH_MIN = 1.3;
const EDGE_WIDTH_MAX = 2.4;
const EDGE_WIDTH_RATIO = 0.0068;
const CONFIDENCE_THRESHOLD = 0.05;
const BG_COLOR = '#f7f4ef';
const EDGE_COLOR = '#8f5f43';
const JOINT_COLOR = '#c98256';
const GUIDE_COLOR = 'rgba(120, 120, 120, 0.12)';
const FRAME_BORDER_COLOR = 'rgba(255, 255, 255, 0.52)';
const FRAME_GLOW_COLOR = 'rgba(18, 18, 18, 0.18)';

type FrameLayout = {
  offsetX: number;
  offsetY: number;
  renderWidth: number;
  renderHeight: number;
  scale: number;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function buildLayout(
  width: number,
  height: number,
  frameWidth: number,
  frameHeight: number,
  padX: number,
  padY: number,
): FrameLayout {
  const scale = Math.min(
    (width - padX * 2) / Math.max(1, frameWidth),
    (height - padY * 2) / Math.max(1, frameHeight),
  );
  const renderWidth = frameWidth * scale;
  const renderHeight = frameHeight * scale;
  return {
    offsetX: (width - renderWidth) * 0.5,
    offsetY: (height - renderHeight) * 0.5,
    renderWidth,
    renderHeight,
    scale,
  };
}

function drawViewport(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  data: Pose2DData,
  frameIndex: number,
  layout: FrameLayout,
  hasVideo: boolean,
): void {
  const frame = data.keypoints2d[frameIndex] ?? data.keypoints2d[0];
  const minSide = Math.min(width, height);
  const edgeWidth = clamp(minSide * EDGE_WIDTH_RATIO, EDGE_WIDTH_MIN, EDGE_WIDTH_MAX);
  const jointRadius = clamp(minSide * JOINT_RADIUS_RATIO, JOINT_RADIUS_MIN, JOINT_RADIUS_MAX);

  if (!hasVideo) {
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = GUIDE_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(width * 0.18, height * 0.82);
    ctx.lineTo(width * 0.82, height * 0.82);
    ctx.moveTo(width * 0.5, height * 0.12);
    ctx.lineTo(width * 0.5, height * 0.92);
    ctx.stroke();
  }

  if (hasVideo) {
    ctx.save();
    ctx.shadowColor = FRAME_GLOW_COLOR;
    ctx.shadowBlur = 18;
    ctx.strokeStyle = FRAME_BORDER_COLOR;
    ctx.lineWidth = 1.2;
    ctx.strokeRect(layout.offsetX + 0.5, layout.offsetY + 0.5, layout.renderWidth - 1, layout.renderHeight - 1);
    ctx.restore();
  }

  const points = frame.map((joint) => ({
    x: layout.offsetX + joint[0] * layout.scale,
    y: layout.offsetY + joint[1] * layout.scale,
    conf: joint[2] ?? 1,
  }));

  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = EDGE_COLOR;
  ctx.lineWidth = edgeWidth;
  data.edges.forEach(([start, end]) => {
    const startPoint = points[start];
    const endPoint = points[end];
    if (!startPoint || !endPoint || startPoint.conf < CONFIDENCE_THRESHOLD || endPoint.conf < CONFIDENCE_THRESHOLD) {
      return;
    }
    ctx.beginPath();
    ctx.moveTo(startPoint.x, startPoint.y);
    ctx.lineTo(endPoint.x, endPoint.y);
    ctx.stroke();
  });

  points.forEach((point) => {
    if (point.conf < CONFIDENCE_THRESHOLD) {
      return;
    }
    ctx.beginPath();
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.arc(point.x, point.y, jointRadius + 0.85, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.fillStyle = JOINT_COLOR;
    ctx.arc(point.x, point.y, jointRadius, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = 'rgba(62, 62, 62, 0.86)';
  ctx.font = '600 12px "SF Pro Display", "PingFang SC", sans-serif';
  ctx.fillText(`frame ${frameIndex + 1}/${data.frame_count}`, 16, 22);
}

export function Pose2DViewport({
  dataUrl,
  currentTime,
  className = '',
  emptyText = '暂无 2D 骨架',
  videoElement = null,
  playing = false,
}: Pose2DViewportProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [data, setData] = useState<Pose2DData | null>(null);
  const [error, setError] = useState('');
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!dataUrl) {
      return;
    }

    const controller = new AbortController();
    fetch(dataUrl, { signal: controller.signal, cache: 'no-store' })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return (await response.json()) as Pose2DData;
      })
      .then((payload) => {
        setData(payload);
        setError('');
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        console.error(reason);
        setData(null);
        setError('2D 骨架读取失败');
      });

    return () => controller.abort();
  }, [dataUrl]);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }
      const nextWidth = Math.max(1, Math.floor(entry.contentRect.width));
      const nextHeight = Math.max(1, Math.floor(entry.contentRect.height));
      setSize((prev) => {
        if (prev.width === nextWidth && prev.height === nextHeight) {
          return prev;
        }
        return { width: nextWidth, height: nextHeight };
      });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const frameIndex = useMemo(() => {
    if (!data || data.frame_count <= 0) {
      return 0;
    }
    return clamp(Math.round(currentTime * data.fps), 0, data.frame_count - 1);
  }, [currentTime, data]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || size.width <= 0 || size.height <= 0) {
      return;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(size.width * dpr);
    canvas.height = Math.round(size.height * dpr);
    canvas.style.width = `${size.width}px`;
    canvas.style.height = `${size.height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.width, size.height);

    const hasVideo = Boolean(videoElement && videoElement.readyState >= 2);
    const layout = buildLayout(
      size.width,
      size.height,
      data.frame_width,
      data.frame_height,
      hasVideo ? VIDEO_INSET_X : EMPTY_PAD_X,
      hasVideo ? VIDEO_INSET_Y : EMPTY_PAD_Y,
    );

    if (hasVideo && videoElement) {
      ctx.drawImage(videoElement, layout.offsetX, layout.offsetY, layout.renderWidth, layout.renderHeight);
    }

    drawViewport(ctx, size.width, size.height, data, frameIndex, layout, hasVideo);
  }, [data, frameIndex, size.height, size.width, videoElement, currentTime, playing]);

  const loading = Boolean(dataUrl) && !data && !error;
  const showPlaceholder = !dataUrl || (!loading && !data);

  return (
    <div
      ref={containerRef}
      className={`relative overflow-hidden rounded-lg border border-zinc-200 bg-[linear-gradient(180deg,#faf7f2_0%,#f1e8dc_100%)] ${className}`.trim()}
    >
      {data ? <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" /> : null}
      {loading ? (
        <div className="absolute inset-0 flex items-center justify-center bg-white/58 text-sm font-medium text-zinc-700">
          正在载入 2D 骨架...
        </div>
      ) : null}
      {showPlaceholder ? (
        <div className="absolute inset-0 flex items-center justify-center px-6 text-center text-sm text-zinc-600">
          {error || emptyText}
        </div>
      ) : null}
    </div>
  );
}
