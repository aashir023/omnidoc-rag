import { FileText, FileType2, FileUp, Loader2, UploadCloud, X } from "lucide-react";
import { useRef, useState } from "react";

function iconFor(type) {
  switch (type) {
    case "pdf":
      return <FileText size={18} className="text-red-400" />;
    case "docx":
      return <FileType2 size={18} className="text-blue-400" />;
    default:
      return <FileUp size={18} className="text-emerald-400" />;
  }
}

function badgeFor(file) {
  switch (file.status) {
    case "uploading":
      return (
        <span className="inline-flex items-center gap-1.5 font-medium text-warning">
          <Loader2 size={12} className="animate-spin" />
          Uploading
        </span>
      );
    case "processing":
      return (
        <span className="inline-flex items-center gap-1.5 font-medium text-accent">
          <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
          Processing
        </span>
      );
    case "error":
      return (
        <span title={file.error} className="inline-flex items-center gap-1.5 font-medium text-error">
          <div className="h-1.5 w-1.5 rounded-full bg-error" />
          Failed
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1.5 font-medium text-success">
          <div className="h-1.5 w-1.5 rounded-full bg-success shadow-[0_0_5px_var(--success)]" />
          Ready
        </span>
      );
  }
}

export default function FileUpload({ onUpload, files, onRemove }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    onUpload(event.dataTransfer.files);
  }

  return (
    <div className="flex flex-col">
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`group relative flex min-h-[160px] w-full flex-col items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed transition-all duration-300 ${
          dragging
            ? "border-accent bg-accent-glow"
            : "border-border bg-surface/30 hover:border-accent/50 hover:bg-elevated/50"
        }`}
      >
        <div className="flex flex-col items-center gap-3">
          <div className={`rounded-full p-3 transition-colors duration-300 ${dragging ? "bg-accent text-white" : "bg-elevated text-accent group-hover:bg-accent group-hover:text-white"}`}>
            <UploadCloud size={24} />
          </div>
          <div>
            <p className="text-sm font-semibold text-primary">Drop files here</p>
            <p className="mt-1 text-[11px] text-muted">PDF · DOCX · TXT supported</p>
          </div>
        </div>
      </button>

      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".pdf,.docx,.txt"
        className="hidden"
        onChange={(event) => onUpload(event.target.files)}
      />

      {files.length > 0 && (
        <div className="mt-6 space-y-2.5">
          {files.map((file) => (
            <div
              key={file.id}
              className="group/card relative flex flex-col gap-2 rounded-xl border border-border bg-elevated/30 p-3 transition-all hover:bg-elevated/60"
            >
              <div className="flex items-center gap-3">
                <div className="shrink-0">{iconFor(file.type)}</div>
                <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-primary" title={file.name}>
                  {file.name}
                </span>
                {(file.status === "ready" || file.status === "error") && (
                  <button
                    type="button"
                    onClick={() => onRemove(file.id)}
                    className="opacity-0 transition-opacity group-hover/card:opacity-100 flex h-6 w-6 items-center justify-center rounded-md bg-background text-secondary hover:text-error"
                  >
                    <X size={14} />
                  </button>
                )}
              </div>
              <div className="flex items-center justify-between text-[10px] uppercase tracking-wider">
                {badgeFor(file)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
