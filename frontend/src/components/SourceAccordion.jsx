import { ChevronDown, FileText, Hash } from "lucide-react";
import { useState } from "react";

export default function SourceAccordion({ sources = [] }) {
  const [open, setOpen] = useState(false);

  if (!sources.length) return null;

  return (
    <div className="flex flex-col">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-secondary transition-colors hover:text-accent"
      >
        <Hash size={12} />
        {sources.length} Reference{sources.length === 1 ? "" : "s"}
        <ChevronDown size={12} className={`transition-transform duration-300 ${open ? "rotate-180" : ""}`} />
      </button>

      <div className={`grid overflow-hidden transition-all duration-500 ease-in-out ${open ? "grid-rows-[1fr] mt-4 opacity-100" : "grid-rows-[0fr] opacity-0"}`}>
        <div className="min-h-0">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {sources.map((source, index) => (
              <div
                key={`${source.source}-${index}`}
                className="group/source relative flex flex-col gap-2 rounded-xl border border-border bg-elevated/40 p-3 transition-all hover:border-accent/30 hover:bg-elevated/60 hover:shadow-lg"
              >
                <div className="flex items-center justify-between gap-2 border-b border-border-subtle pb-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <FileText size={12} className="text-accent" />
                    <span className="truncate text-[10px] font-semibold text-primary" title={source.source}>
                      {source.source}
                    </span>
                  </div>
                  <span className="shrink-0 text-[9px] font-bold text-muted">
                    #{index + 1}
                  </span>
                </div>
                <p className="font-mono text-[11px] leading-relaxed text-secondary line-clamp-4 group-hover/source:line-clamp-none transition-all duration-300">
                  {source.text}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
