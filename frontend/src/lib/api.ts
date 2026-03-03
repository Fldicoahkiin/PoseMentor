import axios from "axios";

export type DatasetItem = {
  id: string;
  name: string;
  stage: string;
  mode: string;
  data_config: string;
  train_config: string;
  notes: string;
};

export type JobItem = {
  job_id: string;
  name: string;
  status: string;
  command: string[];
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  return_code?: number | null;
  error_message?: string | null;
  log_path: string;
};

export type ArtifactStatus = {
  curves_exists: boolean;
  curves_url: string;
  sample_2d_exists: boolean;
  sample_2d_url: string;
  sample_3d_exists: boolean;
  sample_3d_url: string;
  summary_exists: boolean;
  summary_url: string;
};

const baseURL = (import.meta.env.VITE_BACKEND_URL as string | undefined) || "http://127.0.0.1:8787";
export const backendBaseUrl = baseURL.replace(/\/+$/, "");

const client = axios.create({
  baseURL: backendBaseUrl,
  timeout: 15000,
});

export async function fetchHealth() {
  const { data } = await client.get<{ status: string }>("/health");
  return data;
}

export async function fetchDatasets() {
  const { data } = await client.get<{ datasets: DatasetItem[] }>("/datasets");
  return data.datasets;
}

export async function fetchJobs() {
  const { data } = await client.get<{ jobs: JobItem[] }>("/jobs");
  return data.jobs;
}

export async function fetchJobLog(jobId: string) {
  const { data } = await client.get<{ log: string }>(`/jobs/${jobId}/log`);
  return data.log;
}

export async function fetchArtifactStatus() {
  const { data } = await client.get<ArtifactStatus>("/artifacts/status");
  return data;
}

export async function createDataPrepareJob(payload: {
  dataset_id: string;
  config: string;
  download_annotations: boolean;
  extract_annotations: boolean;
  download_videos: boolean;
  video_limit: number;
  agree_license: boolean;
  preprocess_limit: number;
}) {
  const { data } = await client.post<{ job_id: string }>("/jobs/data/prepare", payload);
  return data.job_id;
}

export async function createPoseExtractJob(payload: {
  dataset_id: string;
  config: string;
  input_dir?: string;
  out_dir?: string;
  recursive?: boolean;
  video_ext?: string;
  weights: string;
  conf: number;
  max_videos: number;
}) {
  const { data } = await client.post<{ job_id: string }>("/jobs/pose/extract", payload);
  return data.job_id;
}

export async function createTrainJob(payload: {
  dataset_id: string;
  config: string;
  yolo2d_dir?: string;
  gt3d_dir?: string;
  artifact_dir?: string;
  export_onnx: boolean;
}) {
  const { data } = await client.post<{ job_id: string }>("/jobs/train", payload);
  return data.job_id;
}

export async function createMultiviewJob(payload: {
  config: string;
  limit_sessions: number;
}) {
  const { data } = await client.post<{ job_id: string }>("/jobs/multiview/prepare", payload);
  return data.job_id;
}

export async function createEvaluateJob(payload: {
  dataset_id: string;
  input_dir: string;
  style: string;
  max_videos: number;
  output_csv: string;
}) {
  const { data } = await client.post<{ job_id: string }>("/jobs/evaluate", payload);
  return data.job_id;
}

