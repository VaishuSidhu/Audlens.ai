import { Card } from "@/components/ui/card";
import { Zap, Activity, UploadCloud } from "lucide-react";

const features = [
  {
    icon: Zap,
    title: "Real-time AI detection",
    desc: "Instant deepfake analysis powered by state-of-the-art models.",
  },
  {
    icon: Activity,
    title: "Segment-level analysis",
    desc: "Pinpoint exact timestamps where AI-generated audio appears.",
  },
  {
    icon: UploadCloud,
    title: "Easy audio upload",
    desc: "Drag, drop, and analyze. No setup, no friction.",
  },
];

export function Features() {
  return (
    <section className="container mx-auto px-6 py-24">
      <div className="mx-auto max-w-2xl text-center">
        <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
          Built for <span className="text-gradient">precision</span>
        </h2>
        <p className="mt-3 text-sm text-muted-foreground sm:text-base">
          Everything you need to verify audio authenticity.
        </p>
      </div>

      <div className="mx-auto mt-12 grid max-w-5xl gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {features.map((f) => (
          <Card
            key={f.title}
            className="group border-border/60 bg-card/60 p-6 backdrop-blur transition-all hover:-translate-y-1 hover:border-primary/50 hover:shadow-glow"
          >
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-primary shadow-glow">
              <f.icon className="h-6 w-6 text-primary-foreground" />
            </div>
            <h3 className="text-lg font-semibold">{f.title}</h3>
            <p className="mt-2 text-sm text-muted-foreground">{f.desc}</p>
          </Card>
        ))}
      </div>
    </section>
  );
}
