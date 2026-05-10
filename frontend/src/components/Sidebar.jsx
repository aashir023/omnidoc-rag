import { Trash2, X } from "lucide-react";
import FileUpload from "./FileUpload.jsx";

const LogoMark = ({ className }) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    className={className}
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z"
      className="stroke-accent"
      strokeWidth="2"
    />
    <path
      d="M8 12H16M12 8V16"
      className="stroke-accent"
      strokeWidth="2"
      strokeLinecap="round"
    />
    <path
      d="M12 12L16 16M12 12L8 8"
      className="stroke-accent/50"
      strokeWidth="2"
      strokeLinecap="round"
    />
  </svg>
);

export default function Sidebar({ documents, onClearConversation, isOpen, onClose }) {
  return (
    <aside
      className={`fixed inset-y-0 left-0 z-50 flex w-[300px] flex-col border-r border-border bg-surface/80 bg-gradient-to-b from-surface/50 to-background/50 backdrop-blur-xl transition-transform duration-300 ease-in-out md:translate-x-0 ${
        isOpen ? "translate-x-0" : "-translate-x-full"
      }`}
    >
      <div className="flex flex-col p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent-glow p-2">
              <LogoMark className="h-full w-full" />
            </div>
            <div>
              <h1 className="bg-gradient-accent bg-clip-text text-xl font-bold text-transparent">
                OmniDoc
              </h1>
              <p className="text-[10px] font-medium uppercase tracking-[0.08em] text-secondary">
                Intelligent Analysis
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-border text-secondary transition-colors hover:bg-elevated hover:text-primary md:hidden"
          >
            <X size={18} />
          </button>
        </div>

        <div className="my-8 h-px w-full bg-gradient-to-r from-transparent via-border to-transparent" />

        <div className="flex flex-col">
          <h2 className="mb-4 text-[11px] font-semibold uppercase tracking-[0.1em] text-secondary">
            Knowledge Base
          </h2>
          <FileUpload
            onUpload={documents.uploadFiles}
            files={documents.files}
            onRemove={documents.removeFile}
          />
        </div>

        {documents.activeFiles.length > 0 && (
          <div className="mt-4 flex items-center gap-2 px-1">
            <div className="h-2 w-2 animate-pulse rounded-full bg-success shadow-[0_0_8px_rgba(52,211,153,0.5)]" />
            <span className="text-xs font-medium text-secondary">
              {documents.activeFiles.length} document{documents.activeFiles.length === 1 ? "" : "s"} active
            </span>
          </div>
        )}
      </div>

      <div className="mt-auto flex flex-col p-6">
        <div className="mb-6 h-px w-full bg-border-subtle" />
        
        <button
          type="button"
          onClick={onClearConversation}
          className="group flex w-full items-center justify-center gap-2 rounded-xl border border-border px-4 py-2.5 text-sm font-medium text-secondary transition-all hover:border-error/30 hover:bg-error/5 hover:text-error"
        >
          <Trash2 size={16} className="transition-transform group-hover:scale-110" />
          Clear Conversation
        </button>

        <div className="mt-6 flex flex-col gap-1 text-center">
          <p className="text-[10px] font-medium tracking-wider text-muted">
            ENGINE
          </p>
          <p className="text-[10px] text-muted">
            LLaMA 3.1 · Pinecone · LangChain
          </p>
        </div>
      </div>
    </aside>
  );
}
