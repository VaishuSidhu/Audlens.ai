import { useRef, useState, type DragEvent } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { UploadCloud, FileAudio, X, Sparkles } from "lucide-react";
import { toast } from "sonner";

const ACCEPTED = [".wav", ".mp3", ".ogg"];

export function UploadSection({
  onAnalyze,
  isAnalyzing,
}: {
  onAnalyze: (file: File) => void;
  isAnalyzing: boolean;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File | null | undefined) => {
    if (!f) return;
    const lower = f.name.toLowerCase();
    if (!ACCEPTED.some((ext) => lower.endsWith(ext))) {
      toast.error("Unsupported format", { description: "Please upload a .wav, .mp3, or .ogg file." });
      return;
    }
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setFile(f);
    setAudioUrl(URL.createObjectURL(f));
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files?.[0]);
  };

  const reset = () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setFile(null);
    setAudioUrl(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <section id="upload" className="container mx-auto px-6 py-20">
      <div className="mx-auto max-w-3xl text-center">
        <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
          Upload your <span className="text-gradient">audio</span>
        </h2>
        <p className="mt-3 text-sm text-muted-foreground sm:text-base">
          Drop a .wav, .mp3, or .ogg file below to begin analysis.
        </p>
      </div>

      <Card className="mx-auto mt-10 max-w-3xl border-border/60 bg-card/60 p-6 backdrop-blur sm:p-8">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
          className={`group flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-12 text-center transition-all ${
            dragOver
              ? "border-primary bg-primary/5 shadow-glow"
              : "border-border hover:border-primary/60 hover:bg-primary/5"
          }`}
        >
          <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-gradient-primary shadow-glow">
            <UploadCloud className="h-7 w-7 text-primary-foreground" />
          </div>
          <p className="text-base font-medium text-foreground">Drag & drop your audio file</p>
          <p className="mt-1 text-xs text-muted-foreground">or click to browse · .wav, .mp3, .ogg</p>
          <input
            ref={inputRef}
            type="file"
            accept=".wav,.mp3,.ogg,audio/wav,audio/mpeg,audio/ogg"
            className="hidden"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
        </div>

        {file && audioUrl && (
          <div className="mt-6 space-y-4 rounded-xl border border-border/60 bg-background/40 p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/15">
                  <FileAudio className="h-5 w-5 text-primary-glow" />
                </div>
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{file.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
              <Button variant="ghost" size="icon" onClick={reset} aria-label="Remove file">
                <X className="h-4 w-4" />
              </Button>
            </div>

            <audio controls src={audioUrl} className="w-full" />

            <Button
              onClick={() => onAnalyze(file)}
              disabled={isAnalyzing}
              size="lg"
              className="w-full bg-gradient-primary text-primary-foreground shadow-glow transition-transform hover:scale-[1.01]"
            >
              <Sparkles className="mr-2 h-4 w-4" />
              {isAnalyzing ? "Analyzing..." : "Analyze Audio"}
            </Button>
          </div>
        )}
      </Card>
    </section>
  );
}
