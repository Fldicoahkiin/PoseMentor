import type { PosePreviewAlignment } from '../lib/api';
import { formatDecimal, formatFrameOffset } from '../lib/videoUtils';

type AlignmentInfoPanelProps = {
  alignment: PosePreviewAlignment;
  gridClasses: string;
};

export function AlignmentInfoPanel({ alignment, gridClasses }: AlignmentInfoPanelProps) {
  return (
    <div className="mt-4 rounded-xl border border-zinc-200 bg-stone-50 p-3">
      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-zinc-600">
        <span className="rounded-md border border-zinc-200 bg-white px-2 py-1">
          对齐模式：{alignment.mode === 'aist_official_projection' ? 'AIST 官方时间轴' : alignment.mode}
        </span>
        <span className="rounded-md border border-zinc-200 bg-white px-2 py-1">
          相机方案：{alignment.setting_name}
        </span>
        <span className="rounded-md border border-zinc-200 bg-white px-2 py-1">
          公共时间轴：{formatDecimal(alignment.timeline_fps, 3)} FPS · 起点 {alignment.timeline_start_frame} · {alignment.frame_total} 帧
        </span>
        <span className="rounded-md border border-zinc-200 bg-white px-2 py-1">
          参数文件：{alignment.setting_file.split('/').at(-1) || alignment.setting_file}
        </span>
      </div>
      <div className={`mt-3 ${gridClasses}`}>
        {alignment.available_cameras.map((cameraId) => {
          const geometry = alignment.camera_geometry[cameraId];
          return (
            <div key={cameraId} className="rounded-lg border border-zinc-200 bg-white px-2.5 py-2 text-[11px] leading-5 text-zinc-600 shadow-sm">
              <div className="flex items-center justify-between gap-2">
                <span className="font-semibold text-zinc-900">{cameraId}</span>
                <span className="rounded-md border border-zinc-200 bg-stone-50 px-1.5 py-0.5 text-[10px] text-zinc-600">
                  offset {formatFrameOffset(alignment.camera_offsets[cameraId])}
                </span>
              </div>
              <div className="mt-1.5 space-y-0.5">
                <div>trim {alignment.camera_trim_start[cameraId]} · 误差 {formatDecimal(alignment.camera_sync_error_px[cameraId], 1)} px</div>
                <div>画幅 {geometry?.image_size?.[0] ?? '-'} × {geometry?.image_size?.[1] ?? '-'}</div>
                <div>fx/fy {formatDecimal(geometry?.focal_length_px?.[0], 1)} / {formatDecimal(geometry?.focal_length_px?.[1], 1)}</div>
                <div>cx/cy {formatDecimal(geometry?.principal_point_px?.[0], 1)} / {formatDecimal(geometry?.principal_point_px?.[1], 1)}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
