export default function TypingIndicator() {
  return (
    <div className="flex w-fit items-center gap-1.5 rounded-[4px_18px_18px_18px] border border-border bg-surface px-5 py-4 shadow-sm">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="h-1.5 w-1.5 animate-bounce-subtle rounded-full bg-accent/60"
          style={{ animationDelay: `${i * 150}ms` }}
        />
      ))}
    </div>
  );
}
