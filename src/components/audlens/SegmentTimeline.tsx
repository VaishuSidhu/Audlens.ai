import { Card } from "@/components/ui/card";
import type { Segment } from "@/lib/analyze-audio";

export function SegmentTimeline({ segments, duration }: { segments: Segment[]; duration: number }) {
  const total = Math.max(duration, ...segments.map((s) => s.end), 1);

  return (
    <Card className="border-border/60 bg-card/60 p-6 backdrop-blur sm:p-8">
      <div className="mb-6">
        <h3 className="text-lg font-semibold">Segment Visualization</h3>
        <p className="text-xs text-muted-foreground">
          Highlighted regions indicate detected AI-generated segments.
        </p>
      </div>

      {/* Timeline */}
      <div className="relative">
        <div className="relative h-16 overflow-hidden rounded-lg border border-border/60 bg-background/60">
          {/* Faux waveform */}
          <div className="absolute inset-0 flex items-center justify-around px-1 opacity-40">
            {Array.from({ length: 80 }).map((_, i) => {
              const h = 20 + Math.abs(Math.sin(i * 0.7)) * 70;
              return (
                <span
                  key={i}
                  className="block w-[2px] rounded-full bg-foreground/60"
                  style={{ height: `${h}%` }}
                />
              );
            })}
          </div>

          {/* Fake segment overlays */}
          {segments.map((seg, i) => {
            const left = (seg.start / total) * 100;
            const width = ((seg.end - seg.start) / total) * 100;
            return (
              <div
                key={i}
                title={`${seg.start.toFixed(1)}s – ${seg.end.toFixed(1)}s`}
                className="absolute top-0 bottom-0 border-x border-destructive/80 bg-destructive/40 backdrop-blur-[1px] transition-all hover:bg-destructive/60"
                style={{ left: `${left}%`, width: `${width}%` }}
              />
            );
          })}
        </div>

        <div className="mt-2 flex justify-between text-[10px] text-muted-foreground">
          <span>0.0s</span>
          <span>{total.toFixed(1)}s</span>
        </div>
      </div>

      {/* Segments list */}
      <div className="mt-6">
        <p className="mb-3 text-xs uppercase tracking-wider text-muted-foreground">
          Detected segments ({segments.length})
        </p>

        {segments.length === 0 ? (
          <div className="rounded-lg border border-border/60 bg-background/40 p-4 text-center text-sm text-muted-foreground">
            No fake segments detected.
          </div>
        ) : (
          <ul className="space-y-2">
            {segments.map((seg, i) => (
              <li
                key={i}
                className="flex items-center justify-between rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm"
              >
                <div className="flex items-center gap-3">
                  <span className="inline-flex h-6 min-w-6 items-center justify-center rounded-full bg-destructive/20 px-2 text-xs font-semibold text-destructive">
                    #{i + 1}
                  </span>
                  <span className="font-mono text-foreground">
                    {seg.start.toFixed(1)}s – {seg.end.toFixed(1)}s
                  </span>
                </div>
                {typeof seg.confidence === "number" && (
                  <span className="text-xs text-muted-foreground">
                    {Math.round(seg.confidence * 100)}% confidence
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
}
