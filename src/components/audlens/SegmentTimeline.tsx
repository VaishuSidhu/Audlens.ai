import { Card } from "@/components/ui/card";
import type { Segment } from "@/lib/analyze-audio";

export function SegmentTimeline({ segments, duration }: { segments: Segment[]; duration: number }) {
  const total = Math.max(duration, ...segments.map((s) => s.end), 1);

  return (
    <Card className="border-border/60 bg-card/60 p-6 backdrop-blur sm:p-8">
      <div className="mb-6">
        <h3 className="text-lg font-semibold">Forensic Segment Visualization</h3>
        <p className="text-xs text-muted-foreground">
          Timeline mapping: <span className="text-success font-medium">Green = Human</span> · <span className="text-destructive font-medium">Red = AI/Fake</span> · <span className="text-orange-500 font-medium">Yellow = Suspicious</span>
        </p>
      </div>

      {/* Timeline */}
      <div className="relative">
        <div className="relative h-20 overflow-hidden rounded-lg border border-border/60 bg-background/60 shadow-inner">
          {/* Faux waveform background */}
          <div className="absolute inset-0 flex items-center justify-around px-1 opacity-20">
            {Array.from({ length: 120 }).map((_, i) => {
              const h = 10 + Math.abs(Math.sin(i * 0.4)) * 80;
              return (
                <span
                  key={i}
                  className="block w-[1px] rounded-full bg-foreground/40"
                  style={{ height: `${h}%` }}
                />
              );
            })}
          </div>

          {/* Segment overlays */}
          {segments.map((seg, i) => {
            const left = (seg.start / total) * 100;
            const width = ((seg.end - seg.start) / total) * 100;
            const isFake = seg.prediction === "FAKE";
            const isSuspicious = seg.prediction === "SUSPICIOUS";
            return (
              <div
                key={i}
                title={`${seg.prediction}: ${seg.start.toFixed(1)}s – ${seg.end.toFixed(1)}s`}
                className={`absolute top-0 bottom-0 border-x transition-all hover:brightness-125 ${
                  isFake 
                    ? "border-destructive/60 bg-destructive/30 backdrop-blur-[1px]" 
                    : isSuspicious
                    ? "border-orange-500/60 bg-orange-500/30 backdrop-blur-[1px]"
                    : "border-success/60 bg-success/30 backdrop-blur-[1px]"
                }`}
                style={{ left: `${left}%`, width: `${width}%`, zIndex: isFake ? 10 : isSuspicious ? 8 : 5 }}
              />
            );
          })}
        </div>

        <div className="mt-2 flex justify-between text-[10px] font-medium text-muted-foreground">
          <span>0.0s</span>
          <span>Timeline Scan Range</span>
          <span>{total.toFixed(1)}s</span>
        </div>
      </div>

      {/* Segments list */}
      <div className="mt-8">
        <div className="flex items-center justify-between mb-4">
          <p className="text-xs uppercase tracking-wider font-semibold text-muted-foreground">
            Analysis Timeline ({segments.length} segments)
          </p>
        </div>

        {segments.length === 0 ? (
          <div className="rounded-lg border border-border/60 bg-background/40 p-6 text-center text-sm text-muted-foreground italic">
            No segment data available for this file.
          </div>
        ) : (
          <div className="max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
            <ul className="space-y-2">
              {segments.map((seg, i) => {
                const isFake = seg.prediction === "FAKE";
                const isSuspicious = seg.prediction === "SUSPICIOUS";
                return (
                  <li
                    key={i}
                    className={`flex items-center justify-between rounded-lg border px-4 py-3 text-sm transition-colors ${
                      isFake 
                        ? "border-destructive/30 bg-destructive/5 hover:bg-destructive/10" 
                        : isSuspicious
                        ? "border-orange-500/30 bg-orange-500/5 hover:bg-orange-500/10"
                        : "border-success/30 bg-success/5 hover:bg-success/10"
                    }`}
                  >
                    <div className="flex items-center gap-4">
                      <span className={`inline-flex h-6 min-w-16 items-center justify-center rounded-full px-2 text-[10px] font-bold uppercase tracking-tighter ${
                        isFake ? "bg-destructive/20 text-destructive" : isSuspicious ? "bg-orange-500/20 text-orange-500" : "bg-success/20 text-success"
                      }`}>
                        {seg.prediction}
                      </span>
                      <span className="font-mono text-foreground font-medium">
                        {seg.start.toFixed(2)}s – {seg.end.toFixed(2)}s
                      </span>
                    </div>
                    <div className="flex items-center gap-4">
                      {seg.reason && (
                        <span className="hidden sm:inline text-[11px] text-muted-foreground italic">
                          {seg.reason}
                        </span>
                      )}
                      <span className={`text-xs font-bold ${isFake ? "text-destructive/80" : isSuspicious ? "text-orange-500/80" : "text-success/80"}`}>
                        {Math.round(seg.confidence * 100)}%
                      </span>
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </div>
    </Card>
  );
}
