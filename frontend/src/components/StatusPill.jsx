export default function StatusPill({ activeCount }) {
  const ready = activeCount > 0;

  return (
    <div className="inline-flex items-center gap-2.5 rounded-full border border-border bg-elevated/50 px-3.5 py-1.5 text-xs font-semibold tracking-wide transition-all duration-300">
      <div
        className={`h-2 w-2 rounded-full transition-all duration-500 ${
          ready
            ? "bg-success shadow-[0_0_8px_var(--success)]"
            : "bg-muted"
        }`}
      />
      <span className={ready ? "text-success" : "text-secondary"}>
        {ready ? "Ready to Chat" : "Upload documents to begin"}
      </span>
    </div>
  );
}
