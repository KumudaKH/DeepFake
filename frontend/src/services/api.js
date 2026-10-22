/**
 * API Service Client for the Deepfake Detection & Digital Forensics Backend.
 * Uses configurable base URL via VITE_API_BASE_URL with localhost:8000 fallback.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Compute real SHA-256 hash in browser using native Web Crypto API.
 * Guarantees zero fabricated hashes even if offline.
 */
export async function computeClientSha256(file) {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  } catch (err) {
    console.error('Failed to compute client SHA-256:', err);
    return 'unavailable';
  }
}

/**
 * Health check ping to verify backend connectivity and model checkpoint status.
 */
export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) {
      return { connected: false, error: `HTTP ${res.status}` };
    }
    const data = await res.json();
    return { connected: true, data };
  } catch (err) {
    return {
      connected: false,
      error: err.name === 'TimeoutError' ? 'Connection timed out' : 'Backend offline or unreachable',
    };
  }
}

/**
 * Submit an image file for deepfake classification and Grad-CAM explainability.
 */
export async function analyzeImage(file, threshold = 0.5, alpha = 0.4) {
  const formData = new FormData();
  formData.append('file', file);

  const url = new URL(`${API_BASE_URL}/api/detect/image`);
  if (threshold !== undefined && threshold !== null) {
    url.searchParams.append('threshold', threshold.toString());
  }
  url.searchParams.append('alpha', alpha.toString());

  const res = await fetch(url.toString(), {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errData.detail || `Server error (${res.status})`);
  }

  return await res.json();
}

/**
 * Submit a video file for temporal sampling, face tracking, and logit aggregation.
 */
export async function analyzeVideo(file, maxFrames = 16, threshold = 0.5) {
  const formData = new FormData();
  formData.append('file', file);

  const url = new URL(`${API_BASE_URL}/api/detect/video`);
  url.searchParams.append('max_frames', maxFrames.toString());
  if (threshold !== undefined && threshold !== null) {
    url.searchParams.append('threshold', threshold.toString());
  }

  const res = await fetch(url.toString(), {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errData.detail || `Server error (${res.status})`);
  }

  return await res.json();
}

/**
 * Construct direct printable HTML forensic report URL.
 */
export function getHtmlReportUrl(evidenceId, filename, sha256, verdict, probability, mediaType = 'image', options = {}) {
  const url = new URL(`${API_BASE_URL}/api/report/html/${encodeURIComponent(evidenceId)}`);
  url.searchParams.append('filename', filename || 'evidence');
  url.searchParams.append('sha256', sha256 || '');
  url.searchParams.append('verdict', verdict || 'inconclusive');
  url.searchParams.append('probability', probability !== undefined && probability !== null ? probability.toString() : '0.5');
  url.searchParams.append('media_type', mediaType);
  if (options.isMock !== undefined) {
    url.searchParams.append('is_mock', options.isMock.toString());
  }
  if (options.rawLogit !== undefined && options.rawLogit !== null) {
    url.searchParams.append('raw_logit', options.rawLogit.toString());
  }
  if (options.fileSizeBytes !== undefined && options.fileSizeBytes !== null) {
    url.searchParams.append('file_size_bytes', options.fileSizeBytes.toString());
  }
  if (options.framesCount !== undefined && options.framesCount !== null) {
    url.searchParams.append('frames_count', options.framesCount.toString());
  }
  if (options.examinationTimestampUtc) {
    url.searchParams.append('examination_timestamp_utc', options.examinationTimestampUtc);
  }
  if (options.decisionThreshold !== undefined && options.decisionThreshold !== null) {
    url.searchParams.append('decision_threshold', options.decisionThreshold.toString());
  }
  return url.toString();
}

/**
 * Construct backend JSON forensic report URL.
 */
export function getJsonReportUrl(evidenceId, filename, sha256, verdict, probability, mediaType = 'image', options = {}) {
  const url = new URL(`${API_BASE_URL}/api/report/json/${encodeURIComponent(evidenceId)}`);
  url.searchParams.append('filename', filename || 'evidence');
  url.searchParams.append('sha256', sha256 || '');
  url.searchParams.append('verdict', verdict || 'inconclusive');
  url.searchParams.append('probability', probability !== undefined && probability !== null ? probability.toString() : '0.5');
  url.searchParams.append('media_type', mediaType);
  if (options.isMock !== undefined) {
    url.searchParams.append('is_mock', options.isMock.toString());
  }
  if (options.rawLogit !== undefined && options.rawLogit !== null) {
    url.searchParams.append('raw_logit', options.rawLogit.toString());
  }
  if (options.fileSizeBytes !== undefined && options.fileSizeBytes !== null) {
    url.searchParams.append('file_size_bytes', options.fileSizeBytes.toString());
  }
  if (options.framesCount !== undefined && options.framesCount !== null) {
    url.searchParams.append('frames_count', options.framesCount.toString());
  }
  if (options.examinationTimestampUtc) {
    url.searchParams.append('examination_timestamp_utc', options.examinationTimestampUtc);
  }
  if (options.decisionThreshold !== undefined && options.decisionThreshold !== null) {
    url.searchParams.append('decision_threshold', options.decisionThreshold.toString());
  }
  return url.toString();
}

export { API_BASE_URL };
