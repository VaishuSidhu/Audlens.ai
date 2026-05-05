export type Segment = { 
  start: number; 
  end: number; 
  prediction: "REAL" | "FAKE" | "SUSPICIOUS";
  confidence: number; 
  reason?: string;
};

export type AnalyzeResponse = {
  prediction: "REAL" | "FAKE" | "HYBRID";
  confidence: number; 
  segments: Segment[];
  duration: number;
  forensics?: {
    spectral_consistency: "HIGH" | "LOW";
    detected_fake_points: number[];
    human_confidence: number;
  };
  spectrogram_url?: string;
};

// Configurable endpoint — relative paths for deployment
export const ANALYZE_ENDPOINT = "/analyze-audio";
export const DOWNLOAD_ENDPOINT = "/download-report";

export async function downloadReport(file: File): Promise<void> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(DOWNLOAD_ENDPOINT, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Server returned ${res.status}`);
  
  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `AudLens_Report_${file.name}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}

export async function analyzeAudio(file: File): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(ANALYZE_ENDPOINT, { method: "POST", body: form });
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server returned ${res.status}`);
    }
    const json = (await res.json()) as AnalyzeResponse;
    return normalize(json);
  } catch (error) {
    console.error("Analysis failed:", error);
    throw error;
  }
}

function normalize(r: AnalyzeResponse): AnalyzeResponse {
  const conf = r.confidence > 1 ? r.confidence / 100 : r.confidence;
  return { ...r, confidence: conf };
}

function mockAnalyze(file: File): AnalyzeResponse {
  // Deterministic-ish mock based on filename length.
  const seed = file.name.length;
  const isFake = seed % 2 === 0;
  const duration = 12; // seconds
  const segments: Segment[] = isFake
    ? [
        { start: 2.3, end: 4.1, prediction: "FAKE", confidence: 0.94, reason: "Synthetic artifacts" },
        { start: 6.8, end: 8.2, prediction: "SUSPICIOUS", confidence: 0.88, reason: "Low spectral consistency" },
        { start: 10.1, end: 11.4, prediction: "FAKE", confidence: 0.81, reason: "CNN detection" },
      ]
    : [];
  return {
    prediction: isFake ? "FAKE" : "REAL",
    confidence: isFake ? 0.92 : 0.97,
    segments,
    duration,
    forensics: {
      spectral_consistency: isFake ? "LOW" : "HIGH",
      detected_fake_points: isFake ? [2.3, 6.8, 10.1] : [],
      human_confidence: isFake ? 0.15 : 0.98,
    },
  };
}
