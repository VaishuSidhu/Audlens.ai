import { Button } from "@/components/ui/button";
import { ArrowRight, Waves } from "lucide-react";

export function Hero() {
  const scrollToUpload = () => {
    document.getElementById("upload")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative overflow-hidden">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-40 left-1/2 h-[500px] w-[500px] -translate-x-1/2 rounded-full bg-primary/30 blur-[120px]" />
        <div className="absolute top-40 right-0 h-[400px] w-[400px] rounded-full bg-primary-glow/20 blur-[120px]" />
      </div>

      <div className="container relative mx-auto px-6 pt-28 pb-24 text-center sm:pt-36 sm:pb-32">
        <div className="mx-auto mb-8 inline-flex items-center gap-2 rounded-full border border-border bg-card/60 px-4 py-1.5 text-xs text-muted-foreground backdrop-blur">
          <Waves className="h-3.5 w-3.5 text-primary-glow" />
          Powered by AI Deepfake Detection
        </div>

        <h1 className="text-5xl font-bold tracking-tight sm:text-7xl md:text-8xl">
          <span className="text-gradient">AudLens Audio AI</span>
        </h1>
        <p className="mt-4 text-lg font-medium text-foreground/90 sm:text-xl">
          Advanced Deepfake Detection System
        </p>
        <p className="mx-auto mt-6 max-w-xl text-balance text-sm text-muted-foreground sm:text-base">
          Upload an audio file to detect whether it contains AI-generated fake segments — instantly,
          with segment-level precision.
        </p>

        <div className="mt-10 flex items-center justify-center">
          <Button
            size="lg"
            onClick={scrollToUpload}
            className="bg-gradient-primary text-primary-foreground shadow-glow transition-transform hover:scale-105 hover:shadow-glow"
          >
            Get Started
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </div>
    </section>
  );
}
