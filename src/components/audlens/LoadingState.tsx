import { Card } from "@/components/ui/card";

export function LoadingState() {
  return (
    <Card className="mx-auto max-w-3xl border-border/60 bg-card/60 p-10 text-center backdrop-blur">
      <div className="mx-auto flex h-20 items-end justify-center gap-1.5">
        {[0, 1, 2, 3, 4, 5, 6].map((i) => (
          <span
            key={i}
            className="block w-2 origin-bottom rounded-full bg-gradient-primary animate-pulse-bar"
            style={{ height: "100%", animationDelay: `${i * 0.12}s` }}
          />
        ))}
      </div>
      <p className="mt-6 text-base font-medium text-foreground">Analyzing audio...</p>
      <p className="mt-1 text-xs text-muted-foreground">Running deepfake detection on your file.</p>
    </Card>
  );
}
