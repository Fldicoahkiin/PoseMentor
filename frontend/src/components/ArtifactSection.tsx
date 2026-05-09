import { FileStack, LoaderCircle } from 'lucide-react';
import { Button } from './ui/Button';
import {
  backendBaseUrl,
  type ArtifactManifestItem,
  type ArtifactManifestPayload,
  type ArtifactStatus,
} from '../lib/api';
import { formatBytes, formatTime } from '../lib/videoUtils';

interface ArtifactSectionProps {
  artifactManifest: ArtifactManifestPayload | null;
  artifactStatus: ArtifactStatus | null;
  modelFiles: ArtifactManifestItem[];
  reportFiles: ArtifactManifestItem[];
  summaryText: string;
}

export function ArtifactSection({
  artifactManifest,
  artifactStatus,
  modelFiles,
  reportFiles,
  summaryText,
}: ArtifactSectionProps) {
  return (
    <section className="grid grid-cols-1 gap-6 2xl:grid-cols-2">
      <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="flex items-center gap-2 text-base font-bold text-zinc-800">
            <FileStack size={18} />
            模型产物
          </h2>
          <span className="text-xs text-zinc-500">总计 {artifactManifest?.count || 0}</span>
        </div>
        <div className="mb-3 flex flex-wrap gap-2">
          {Object.entries(artifactManifest?.by_kind || {}).map(([kind, count]) => (
            <span key={kind} className="rounded-full border border-zinc-200 bg-stone-50 px-2 py-1 text-xs text-zinc-600">
              {kind}: {count}
            </span>
          ))}
        </div>
        <div className="space-y-2">
          {modelFiles.length > 0 ? (
            modelFiles.map((item) => (
              <div key={item.path} className="flex items-center justify-between rounded-lg border border-zinc-200 bg-stone-50 px-3 py-2">
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-zinc-800">{item.name}</p>
                  <p className="text-xs text-zinc-500">{formatTime(item.updated_at)}</p>
                </div>
                <a
                  href={`${backendBaseUrl}${item.url}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="rounded-md border border-zinc-200 px-2 py-1 text-xs font-semibold text-zinc-700 hover:bg-zinc-100"
                >
                  打开
                </a>
              </div>
            ))
          ) : (
            <div className="flex h-28 items-center justify-center rounded-xl border border-dashed border-zinc-300 bg-stone-50 text-sm text-zinc-500">
              暂无模型文件
            </div>
          )}
        </div>
      </div>

      <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="flex items-center gap-2 text-base font-bold text-zinc-800">
            <LoaderCircle size={18} />
            报告与摘要
          </h2>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              if (artifactStatus?.summary_exists) {
                window.open(`${backendBaseUrl}${artifactStatus.summary_url}`, '_blank', 'noopener,noreferrer');
              }
            }}
            disabled={!artifactStatus?.summary_exists}
          >
            打开摘要
          </Button>
        </div>
        <pre className="mb-3 h-28 overflow-auto rounded-lg border border-zinc-200 bg-zinc-950 p-3 text-xs leading-5 text-zinc-200">
          {summaryText || '暂无训练摘要'}
        </pre>
        <div className="max-h-52 space-y-2 overflow-auto">
          {reportFiles.length > 0 ? (
            reportFiles.map((item) => (
              <div key={item.path} className="flex items-center justify-between rounded-lg border border-zinc-200 bg-stone-50 px-3 py-2">
                <div className="min-w-0">
                  <p className="truncate text-sm text-zinc-800">{item.path}</p>
                  <p className="text-xs text-zinc-500">
                    {item.kind} · {formatBytes(item.size_bytes)}
                  </p>
                </div>
                <a
                  href={`${backendBaseUrl}${item.url}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="rounded-md border border-zinc-200 px-2 py-1 text-xs font-semibold text-zinc-700 hover:bg-zinc-100"
                >
                  打开
                </a>
              </div>
            ))
          ) : (
            <div className="flex h-20 items-center justify-center rounded-xl border border-dashed border-zinc-300 bg-stone-50 text-sm text-zinc-500">
              暂无报告文件
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
