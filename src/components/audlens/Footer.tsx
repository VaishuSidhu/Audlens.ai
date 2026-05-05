export function Footer() {
  return (
    <footer className="border-t border-border/60 bg-background/60 py-10 backdrop-blur">
      <div className="container mx-auto px-6 text-center">
        <p className="text-sm font-medium">
          <span className="text-gradient">AudLens Audio AI</span> – Audio Deepfake Detection System
        </p>
        <p className="mt-2 text-xs text-muted-foreground">
          Verifying audio authenticity with AI · © {new Date().getFullYear()} AudLens Audio AI
        </p>
      </div>
    </footer>
  );
}
