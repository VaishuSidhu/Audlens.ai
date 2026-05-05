import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Hero } from "@/components/audlens/Hero";
import { UploadSection } from "@/components/audlens/UploadSection";
import { LoadingState } from "@/components/audlens/LoadingState";
import { ResultCard } from "@/components/audlens/ResultCard";
import { SegmentTimeline } from "@/components/audlens/SegmentTimeline";
import { Features } from "@/components/audlens/Features";
import { Footer } from "@/components/audlens/Footer";
import { Toaster, toast } from "@/components/ui/sonner";
import { analyzeAudio, type AnalyzeResponse } from "@/lib/analyze-audio";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "AudLens Audio AI – Audio Deepfake Detection System" },
      {
        name: "description",
        content:
          "Upload an audio file and detect AI-generated deepfake segments with segment-level precision.",
      },
      { property: "og:title", content: "AudLens Audio AI – Audio Deepfake Detection" },
      {
        property: "og:description",
        content: "Detect AI-generated fake segments in any audio file, instantly.",
      },
    ],
  }),
  component: Index,
});

function Index() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [duration, setDuration] = useState(12);

  // Force dark theme
  useEffect(() => {
    document.documentElement.classList.add("dark");
  }, []);

  const handleAnalyze = async (file: File) => {
    setIsAnalyzing(true);
    setResult(null);
    setCurrentFile(file);

    // Try to read true duration for the timeline.
    try {
      const url = URL.createObjectURL(file);
      const audio = new Audio(url);
      await Promise.race([
        new Promise<void>((resolve) => {
          audio.addEventListener("loadedmetadata", () => {
            if (Number.isFinite(audio.duration) && audio.duration > 0) {
              setDuration(audio.duration);
            }
            resolve();
          });
          audio.addEventListener("error", () => resolve());
        }),
        new Promise<void>((resolve) => setTimeout(resolve, 1000)), // 1s timeout
      ]).finally(() => {
        URL.revokeObjectURL(url);
      });
    } catch {
      /* ignore */
    }

    try {
      const res = await analyzeAudio(file);
      setResult(res);
    } catch (error: any) {
      console.error("Analysis error:", error);
      toast.error("Analysis failed", { 
        description: error.message || "Could not connect to the detection server. Please ensure the backend is running." 
      });
      setResult(null);
    } finally {
      setIsAnalyzing(false);
    }

    setTimeout(() => {
      document.getElementById("results")?.scrollIntoView({ behavior: "smooth" });
    }, 100);
  };

  return (
    <div className="min-h-screen">
      <Hero />
      <UploadSection onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} />

      {(isAnalyzing || result) && (
        <section id="results" className="container mx-auto px-6 pb-20">
          <div className="mx-auto max-w-4xl space-y-6">
            {isAnalyzing && <LoadingState />}
            {!isAnalyzing && result && (
              <>
                <ResultCard result={result} file={currentFile} />
                <SegmentTimeline segments={result.segments} duration={duration} />
              </>
            )}
          </div>
        </section>
      )}

      <Features />
      <Footer />
      <Toaster />
    </div>
  );
}
