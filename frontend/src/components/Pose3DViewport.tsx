import { Canvas } from '@react-three/fiber';
import { Line, OrbitControls } from '@react-three/drei';
import { useEffect, useMemo, useRef, useState } from 'react';

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

type OrbitControlsHandle = {
  target: {
    set: (x: number, y: number, z: number) => void;
  };
  update: () => void;
};

const BG_TOP = '#f7f3eb';
const JOINT_COLOR = '#c88056';
const ROOT_COLOR = '#b16043';
const EDGE_COLOR = '#7a5b4a';
const GRID_COLOR = '#d8cdbc';
const GRID_CENTER_COLOR = '#bfae96';
const LIGHT_COLOR = '#fff8ef';

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function SkeletonFrame({ data, frameIndex }: { data: Pose3DData; frameIndex: number }) {
  const frame = data.joints3d[frameIndex] ?? data.joints3d[0] ?? [];
  const radius = Math.max(data.bounds.max_radius, 0.8);
  const jointRadius = Math.max(radius * 0.03, 0.025);
  const axisSize = Math.max(radius * 0.75, 0.55);
  const gridSize = Math.max(radius * 3.2, 2.4);

  return (
    <group>
      <ambientLight intensity={0.92} color={LIGHT_COLOR} />
      <directionalLight position={[radius * 1.6, radius * 2.2, radius * 1.4]} intensity={1.28} color="#fff5e9" />
      <directionalLight position={[-radius * 1.6, radius * 1.2, -radius * 1.4]} intensity={0.46} color="#f4efe8" />
      <gridHelper args={[gridSize, 16, GRID_CENTER_COLOR, GRID_COLOR]} position={[0, data.bounds.floor_y, 0]} />
      <axesHelper args={[axisSize]} />
      {data.edges.map(([start, end], index) => {
        const startPoint = frame[start];
        const endPoint = frame[end];
        if (!startPoint || !endPoint) {
          return null;
        }
        return (
          <Line
            key={`${start}-${end}-${index}`}
            points={[
              [startPoint[0], startPoint[1], startPoint[2]],
              [endPoint[0], endPoint[1], endPoint[2]],
            ]}
            color={EDGE_COLOR}
            lineWidth={2.4}
          />
        );
      })}
      {frame.map((joint, index) => (
        <mesh key={`joint-${index}`} position={[joint[0], joint[1], joint[2]]} castShadow receiveShadow>
          <sphereGeometry args={[jointRadius, 18, 18]} />
          <meshStandardMaterial color={index === 0 ? ROOT_COLOR : JOINT_COLOR} roughness={0.42} metalness={0.08} />
        </mesh>
      ))}
    </group>
  );
}

function SceneRig({ radius, targetY }: { radius: number; targetY: number }) {
  const controlsRef = useRef<OrbitControlsHandle | null>(null);

  return (
    <OrbitControls
      ref={(controls) => {
        controlsRef.current = controls as OrbitControlsHandle | null;
      }}
      target={[0, targetY, 0]}
      enablePan
      enableZoom
      enableRotate
      enableDamping
      dampingFactor={0.08}
      minDistance={Math.max(radius * 0.9, 0.75)}
      maxDistance={Math.max(radius * 4.4, 3.2)}
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

  const radius = useMemo(() => Math.max(data?.bounds.max_radius ?? 1.0, 0.8), [data]);
  const targetY = useMemo(() => Math.max(radius * 0.2, 0.18), [radius]);
  const sceneKey = `${dataUrl || 'pose-3d-empty'}-${Math.round(radius * 100)}`;
  const cameraPosition = useMemo<[number, number, number]>(
    () => [radius * 1.55, Math.max(radius * 1.18, 0.96), radius * 2.2],
    [radius],
  );
  const cameraFar = useMemo(() => Math.max(radius * 12, 1200), [radius]);
  const showPlaceholder = !dataUrl || (!loading && !data);

  return (
    <div
      className={`relative overflow-hidden rounded-lg border border-zinc-200 bg-[linear-gradient(180deg,#f7f3eb_0%,#ece1d4_100%)] ${className}`.trim()}
    >
      {data ? (
        <Canvas key={sceneKey} dpr={[1, 2]} camera={{ fov: 34, near: 0.1, far: cameraFar, position: cameraPosition }}>
          <color attach="background" args={[BG_TOP]} />
          <fog attach="fog" args={[BG_TOP, radius * 4.4, radius * 7.8]} />
          <SceneRig radius={radius} targetY={targetY} />
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
