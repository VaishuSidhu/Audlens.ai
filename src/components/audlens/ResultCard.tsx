import { useState } from "react";
import { Card } from "@/components/ui/card";
import { CheckCircle2, AlertTriangle, Download, Loader2 } from "lucide-react";
import { type AnalyzeResponse, downloadReport } from "@/lib/analyze-audio";
import { Button } from "@/components/ui/button";

export function ResultCard({ result, file }: { result: AnalyzeResponse, file: File | null }) {
  const isFake = result.prediction === "FAKE";
  const isHybrid = result.prediction === "HYBRID";
  const isSuspicious = result.prediction === "SUSPICIOUS";
  const isReal = result.prediction === "REAL";
  
  const pct = Math.round(result.confidence * 100);
  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownload = async () => {
    if (!file) return;
    setIsDownloading(true);
    try {
      await downloadReport(file);
    } catch (e) {
      console.error("Failed to download report", e);
    } finally {
      setIsDownloading(false);
    }
  };

  const getStatusColor = () => {
    if (isFake) return "text-destructive";
    if (isHybrid) return "text-warning";
    if (isSuspicious) return "text-orange-500";
    return "text-success";
  };

  const getStatusBg = () => {
    if (isFake) return "bg-destructive/15";
    if (isHybrid) return "bg-warning/15";
    if (isSuspicious) return "bg-orange-500/15";
    return "bg-success/15";
  };

  return (
    <Card className="border-border/60 bg-card/60 p-6 backdrop-blur sm:p-8">
      <div className="flex flex-col items-center gap-6 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-4">
          <div
            className={`flex h-14 w-14 items-center justify-center rounded-full ${getStatusBg()}`}
          >
            {isReal ? (
              <CheckCircle2 className="h-7 w-7 text-success" />
            ) : (
              <AlertTriangle className={`h-7 w-7 ${getStatusColor()}`} />
            )}
          </div>
          <div>
            <p className="text-xs uppercase tracking-wider text-muted-foreground">Global Verdict</p>
            <p className={`text-3xl font-bold tracking-tight ${getStatusColor()}`}>
              {isFake ? "FULL FAKE" : isHybrid ? "HYBRID" : isSuspicious ? "SUSPICIOUS" : "REAL"}
            </p>
          </div>
        </div>

        <div className="w-full sm:max-w-xs">
          <div className="mb-2 flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Detection Confidence</span>
            <span className="font-semibold">{pct}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
            <div
              className={`h-full rounded-full transition-all duration-700 ${
                isFake ? "bg-destructive" : isHybrid ? "bg-warning" : isSuspicious ? "bg-orange-500" : "bg-success"
              }`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      </div>
      
      {result.forensics && (
        <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div className="rounded-lg border border-border/40 bg-muted/20 p-4">
            <p className="mb-1 text-xs font-semibold uppercase text-muted-foreground">Forensic Insight</p>
            <p className="text-sm">
              Spectral Consistency: <span className={result.forensics.spectral_consistency === "LOW" ? "font-bold text-destructive" : "font-bold text-success"}>
                {result.forensics.spectral_consistency}
              </span>
            </p>
            <p className="mt-2 text-[10px] leading-tight text-muted-foreground">
              {result.forensics.spectral_consistency === "LOW" 
                ? "Irregular frequency patterns detected, matching typical AI signatures or deepfake artifacts."
                : "Natural frequency variations detected, consistent with human biological vocal traits."}
            </p>
          </div>
          <div className="rounded-lg border border-border/40 bg-muted/20 p-4">
             <p className="mb-1 text-xs font-semibold uppercase text-muted-foreground">Neural Analysis</p>
             <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[11px]">
                  <span>Human Vocal Probability</span>
                  <span>{Math.round(result.forensics.human_confidence * 100)}%</span>
                </div>
                <div className="flex items-center justify-between text-[11px]">
                  <span>Anomalous Segments</span>
                  <span>{result.forensics.detected_fake_points.length}</span>
                </div>
             </div>
          </div>
        </div>
      )}

      <div className="mt-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-t border-border/40 pt-6">
        <p className="text-sm text-muted-foreground">
          {isReal
            ? "No AI-generated segments were detected. This audio appears authentic."
            : isHybrid 
            ? "Caution: This audio contains both human and AI-generated segments. Forensic scan identified hybrid content."
            : isSuspicious
            ? "Notice: Some segments showed unusual characteristics, but no definitive AI artifacts were found. Use caution."
            : "WARNING: FULL FAKE AUDIO BY AI AGENT VOICE"}
        </p>

        {file && (
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handleDownload} 
            disabled={isDownloading}
            className="shrink-0"
          >
            {isDownloading ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Download className="mr-2 h-4 w-4" />
            )}
            Download PDF Report
          </Button>
        )}
      </div>
    </Card>
  );
}
