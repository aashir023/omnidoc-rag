import { SendHorizontal, AlertCircle } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import MessageBubble from "./MessageBubble.jsx";
import StatusPill from "./StatusPill.jsx";
import TypingIndicator from "./TypingIndicator.jsx";

const examples = [
  "Summarize the key points",
  "What are the main conclusions?",
  "List all action items",
];

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

export default function ChatArea({ chat, activeCount }) {
  const [input, setInput] = useState("");
  const scrollRef = useRef(null);
  const textRef = useRef(null);
  const canSend = input.trim().length > 0 && activeCount > 0 && !chat.isStreaming;

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chat.messages, chat.isStreaming]);

  useEffect(() => {
    if (!textRef.current) return;
    textRef.current.style.height = "auto";
    textRef.current.style.height = `${Math.min(textRef.current.scrollHeight, 160)}px`;
  }, [input]);

  function submit(value = input) {
    if (!value.trim() || activeCount === 0 || chat.isStreaming) return;
    chat.sendMessage(value);
    setInput("");
  }

  return (
    <main className="relative flex min-h-screen flex-col md:ml-[300px]">
      <header className="sticky top-0 z-40 flex h-16 items-center justify-between border-b border-border bg-background/80 px-6 backdrop-blur-xl">
        <div className="flex items-center gap-4">
          <h1 className="text-sm font-semibold tracking-tight text-primary sm:text-base">
            Chat with your Documents
          </h1>
        </div>
        <StatusPill activeCount={activeCount} />
      </header>

      <section
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-10 scrollbar-thin sm:px-8"
      >
        {chat.messages.length === 0 ? (
          <div className="flex h-full min-h-[calc(100vh-250px)] animate-fade-up items-center justify-center">
            <div className="max-w-xl text-center">
              <div className="relative mx-auto mb-8 h-20 w-20">
                <div className="absolute inset-0 animate-pulse rounded-3xl bg-accent/20 blur-2xl" />
                <div className="relative flex h-full w-full items-center justify-center rounded-3xl border border-border bg-elevated/50 shadow-2xl backdrop-blur-sm">
                  <LogoMark className="h-10 w-10" />
                </div>
              </div>
              <h2 className="bg-gradient-accent bg-clip-text text-3xl font-bold tracking-tight text-transparent sm:text-4xl">
                Welcome to OmniDoc
              </h2>
              <p className="mt-4 text-base text-secondary">
                Upload your documents and ask anything about them.
              </p>
              <div className="mt-10 flex flex-wrap justify-center gap-3">
                {examples.map((example) => (
                  <button
                    key={example}
                    type="button"
                    onClick={() => submit(example)}
                    disabled={activeCount === 0}
                    className="rounded-full border border-border bg-elevated/40 px-5 py-2.5 text-sm font-medium text-primary transition-all hover:border-accent hover:bg-elevated hover:shadow-[0_0_15px_rgba(79,142,247,0.1)] disabled:cursor-not-allowed disabled:opacity-30"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-4xl space-y-8">
            {chat.messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onRetry={chat.retryLastMessage}
              />
            ))}
            {chat.isStreaming && chat.messages.at(-1)?.role !== "assistant" && (
              <TypingIndicator />
            )}
          </div>
        )}
      </section>

      <footer className="sticky bottom-0 border-t border-border bg-background/95 p-4 backdrop-blur-md sm:p-6">
        <div className="mx-auto max-w-4xl">
          {activeCount === 0 && (
            <div className="mb-4 flex items-center gap-2 rounded-xl bg-warning/5 px-4 py-2 text-xs font-medium text-warning border border-warning/10 animate-fade-in">
              <AlertCircle size={14} />
              Upload documents to start chatting
            </div>
          )}
          
          <div className="group relative flex items-end gap-3 rounded-2xl border border-border bg-elevated/40 p-2 transition-all duration-300 focus-within:border-accent/50 focus-within:bg-elevated/60 focus-within:shadow-[0_0_20px_var(--accent-glow)]">
            <textarea
              ref={textRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (canSend) submit();
                }
              }}
              placeholder="Ask anything about your documents..."
              className="max-h-40 min-h-[44px] flex-1 resize-none bg-transparent px-4 py-3 text-[15px] text-primary outline-none placeholder:text-muted"
            />
            <button
              type="button"
              onClick={() => submit()}
              disabled={!canSend}
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-gradient-accent text-white shadow-lg transition-all hover:scale-105 hover:shadow-accent/25 active:scale-95 disabled:cursor-not-allowed disabled:bg-elevated disabled:text-muted disabled:shadow-none disabled:opacity-50"
            >
              <SendHorizontal size={20} />
            </button>
          </div>
          <p className="mt-3 text-center text-[10px] font-medium tracking-wide text-muted uppercase">
            Press Enter to send · Shift+Enter for new line
          </p>
        </div>
      </footer>
    </main>
  );
}
