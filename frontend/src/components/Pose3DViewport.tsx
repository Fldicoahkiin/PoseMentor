import { Canvas } from '@react-three/fiber';
import { Line, OrbitControls } from '@react-three/drei';
import { useEffect, useMemo, useState } from 'react';

type Pose3DBounds = {
  min: [number, number, number];
  max: [number, number, number];
  floor_y: number;
  max_radius: number;
};

type Pose3DData = {
  fps: number;
  frame_count: number;
  joint_count: number;
  edges: Array<[number, number]>;
  bounds: Pose3DBounds;
  joints3d: number[][][];
};

type Pose3DViewportProps = {
  dataUrl: string;
  currentTime: number;
  playing: boolean;
  className?: string;
  emptyText?: string;
};

const BG_TOP = '#f7f3eb';
const JOINT_COLOR = '#c88056';
const ROOT_COLOR = '#b16043';
const EDGE_COLOR = '#7a5b4a';
const GRID_COLOR = '#d8cdbc';
const GRID_CENTER_COLOR = '#bfae96';
const LIGHT_COLOR = '#fff8ef';
const JOINT_RADIUS = 0.031;
const ROOT_RADIUS = 0.039;
const EDGE_WIDTH = 1.8;
const AXIS_SIZE = 0.74;
const GRID_SIZE = 3.4;
const CAMERA_POSITION: [number, number, number] = [1.56, 1.08, 2.08];
const CAMERA_FAR = 36;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function normalizeJoint(joint: number[], scale: number): [number, number, number] {
  return [joint[0] / scale, joint[1] / scale, joint[2] / scale];
}

function SkeletonFrame({ data, frameIndex }: { data: Pose3DData; frameIndex: number }) {
  const frame = data.joints3d[frameIndex] ?? data.joints3d[0] ?? [];
  const sceneScale = Math.max(data.bounds.max_radius, 1);
  const floorY = data.bounds.floor_y / sceneScale;
  const points = frame.map((joint) => normalizeJoint(joint, sceneScale));

  return (
    <group>
      <ambientLight intensity={0.92} color={LIGHT_COLOR} />
      <directionalLight position={[1.92, 2.4, 1.76]} intensity={1.12} color="#fff5e9" />
      <directionalLight position={[-1.58, 1.32, -1.48]} intensity={0.4} color="#f4efe8" />
      <gridHelper args={[GRID_SIZE, 18, GRID_CENTER_COLOR, GRID_COLOR]} position={[0, floorY, 0]} />
      <axesHelper args={[AXIS_SIZE]} />
      {data.edges.map(([start, end], index) => {
        const startPoint = points[start];
        const endPoint = points[end];
        if (!startPoint || !endPoint) {
          return null;
        }
        return (
          <Line
            key={`${start}-${end}-${index}`}
            points={[startPoint, endPoint]}
            color={EDGE_COLOR}
            lineWidth={EDGE_WIDTH}
          />
        );
      })}
      {points.map((joint, index) => (
        <mesh key={`joint-${index}`} position={joint} castShadow receiveShadow>
          <sphereGeometry args={[index === 0 ? ROOT_RADIUS : JOINT_RADIUS, 16, 16]} />
          <meshStandardMaterial color={index === 0 ? ROOT_COLOR : JOINT_COLOR} roughness={0.46} metalness={0.06} />
        </mesh>
      ))}
    </group>
  );
}

function SceneRig({ targetY }: { targetY: number }) {
  return (
    <OrbitControls
      target={[0, targetY, 0]}
      enablePan
      enableZoom
      enableRotate
      enableDamping
      dampingFactor={0.08}
      minDistance={0.82}
      maxDistance={4.8}
      maxPolarAngle={Math.PI / 2.02}
    />
  );
}

export function Pose3DViewport({ dataUrl, currentTime, playing, className = '', emptyText = '暂无 3D 骨架' }: Pose3DViewportProps) {
  const [data, setData] = useState<Pose3DData | null>(null);
  const [loading, setLoading] = useState(Boolean(dataUrl));
  const [error, setError] = useState('');

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
        return (await response.json()) as Pose3DData;
      })
      .then((payload) => {
        setData(payload);
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        console.error(reason);
        setData(null);
        setError('3D 骨架读取失败');
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      });

    return () => controller.abort();
  }, [dataUrl]);

  const frameIndex = useMemo(() => {
    if (!data || data.frame_count <= 0) {
      return 0;
    }
    return clamp(Math.round(currentTime * data.fps), 0, data.frame_count - 1);
  }, [currentTime, data]);

  const sceneScale = useMemo(() => Math.max(data?.bounds.max_radius ?? 1.0, 1.0), [data]);
  const targetY = useMemo(
    () => clamp(-((data?.bounds.floor_y ?? -1) / sceneScale) * 0.12, 0.06, 0.18),
    [data, sceneScale],
  );
  const sceneKey = `${dataUrl || 'pose-3d-empty'}-${Math.round(sceneScale * 100)}`;
  const showPlaceholder = !dataUrl || (!loading && !data);

  return (
    <div
      className={`relative overflow-hidden rounded-lg border border-zinc-200 bg-[linear-gradient(180deg,#f7f3eb_0%,#ece1d4_100%)] ${className}`.trim()}
    >
      {data ? (
        <Canvas key={sceneKey} dpr={[1, 2]} camera={{ fov: 34, near: 0.1, far: CAMERA_FAR, position: CAMERA_POSITION }}>
          <color attach="background" args={[BG_TOP]} />
          <fog attach="fog" args={[BG_TOP, 4.2, 7.6]} />
          <SceneRig targetY={targetY} />
          <SkeletonFrame data={data} frameIndex={frameIndex} />
        </Canvas>
      ) : null}

      <div className="pointer-events-none absolute left-3 top-3 rounded-lg border border-white/70 bg-white/80 px-3 py-2 text-xs text-zinc-700 shadow-sm backdrop-blur-sm">
        <div className="font-semibold text-zinc-800">3D Skeleton</div>
        <div>
          frame {data ? `${frameIndex + 1}/${data.frame_count}` : '--'} · {playing ? '播放中' : '暂停'}
        </div>
      </div>
      <div className="pointer-events-none absolute right-3 top-3 rounded-lg border border-white/70 bg-white/78 px-3 py-2 text-[11px] text-zinc-600 shadow-sm backdrop-blur-sm">
        拖拽旋转 · 滚轮缩放
      </div>
      <div className="pointer-events-none absolute bottom-3 right-3 rounded-lg border border-white/70 bg-white/76 px-3 py-1.5 text-[11px] text-zinc-600 shadow-sm backdrop-blur-sm">
        XYZ 轴与地面网格已对齐
      </div>

      {loading ? (
        <div className="absolute inset-0 flex items-center justify-center bg-white/56 text-sm font-medium text-zinc-700">
          正在载入 3D 骨架...
        </div>
      ) : null}
      {showPlaceholder ? (
        <div className="absolute inset-0 flex items-center justify-center px-6 text-center text-sm text-zinc-600">
          {error || emptyText}
        </div>
      ) : null}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-20 bg-[linear-gradient(180deg,transparent_0%,rgba(236,225,212,0.76)_100%)]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-16 bg-[linear-gradient(180deg,rgba(247,243,235,0.92)_0%,transparent_100%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_32%,rgba(255,255,255,0.24)_0%,transparent_56%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(135deg,rgba(255,255,255,0.04)_0%,transparent_42%,rgba(0,0,0,0.03)_100%)]" />
      <div className="pointer-events-none absolute inset-0 shadow-[inset_0_1px_0_rgba(255,255,255,0.28)]" />
      <div className="pointer-events-none absolute inset-0 rounded-lg border border-black/4" />
    </div>
  );
}
