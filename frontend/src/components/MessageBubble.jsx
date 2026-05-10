import { User, RotateCcw } from "lucide-react";
import ReactMarkdown from "react-markdown";
import SourceAccordion from "./SourceAccordion.jsx";

const BotAvatar = () => (
  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent-glow border border-accent/20">
    <svg
      viewBox="0 0 24 24"
      fill="none"
      className="h-5 w-5"
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
    </svg>
  </div>
);

const UserAvatar = () => (
  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-elevated border border-border">
    <User size={16} className="text-secondary" />
  </div>
);

export default function MessageBubble({ message, onRetry }) {
  const isUser = message.role === "user";

  return (
    <div className={`group flex gap-4 animate-fade-in ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      <div className="flex flex-col items-center">
        {isUser ? <UserAvatar /> : <BotAvatar />}
      </div>

      <div className={`flex max-w-[80%] flex-col gap-2 ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`relative px-5 py-3.5 shadow-sm transition-all duration-300 ${
            isUser
              ? "rounded-[18px_18px_4px_18px] bg-gradient-accent text-white"
              : "rounded-[4px_18px_18px_18px] border border-border bg-surface text-primary border-l-[3px] border-l-accent"
          }`}
        >
          {message.content ? (
            <div className="prose prose-invert max-w-none">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          ) : !isUser && !message.error ? (
            <div className="flex items-center gap-1.5 py-2">
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  className="h-1.5 w-1.5 animate-bounce-subtle rounded-full bg-accent/60"
                  style={{ animationDelay: `${i * 150}ms` }}
                />
              ))}
            </div>
          ) : null}

          {message.error && (
            <div className="mt-4 rounded-xl border border-error/20 bg-error/5 p-4 text-sm text-error">
              <p className="font-medium">{message.error}</p>
              <button
                type="button"
                onClick={onRetry}
                className="mt-3 flex items-center gap-2 rounded-lg border border-error/30 bg-background/50 px-3 py-1.5 text-xs font-semibold text-primary transition-all hover:bg-error/10 hover:text-error"
              >
                <RotateCcw size={13} />
                Retry Request
              </button>
            </div>
          )}
        </div>
        
        <div className="flex items-center gap-3 px-1">
          <span className="text-[10px] font-medium uppercase tracking-wider text-muted">
            {message.timestamp}
          </span>
          {!isUser && message.sources?.length > 0 && (
            <span className="h-1 w-1 rounded-full bg-border" />
          )}
          {!isUser && <SourceAccordion sources={message.sources} />}
        </div>
      </div>
    </div>
  );
}
