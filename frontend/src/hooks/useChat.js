import { useRef, useState } from "react";

const now = () => new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

export default function useChat(activeFiles, addToast) {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const lastQueryRef = useRef("");

  async function sendMessage(queryOverride) {
    const query = (queryOverride || "").trim();
    if (!query || activeFiles.length === 0 || isStreaming) {
      return;
    }

    lastQueryRef.current = query;
    const assistantId = crypto.randomUUID();
    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
      timestamp: now(),
    };
    const assistantMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      sources: [],
      timestamp: now(),
      error: "",
    };

    setMessages((current) => [...current, userMessage, assistantMessage]);
    setIsStreaming(true);

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, active_files: activeFiles }),
      });

      if (!response.ok || !response.body) {
        throw new Error("The chat service did not respond.");
      }

      await readSseStream(response.body, (event) => {
        if (event.token) {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId ? { ...message, content: message.content + event.token } : message
            )
          );
        }

        if (event.error) {
          throw new Error(event.error);
        }

        if (event.done && event.sources) {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId ? { ...message, sources: event.sources } : message
            )
          );
        }
      });
    } catch (error) {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                error: error.message || "Something went wrong.",
                content: message.content || "I could not complete that request.",
              }
            : message
        )
      );
      addToast("error", error.message || "Chat failed.");
    } finally {
      setIsStreaming(false);
    }
  }

  function retryLastMessage() {
    if (lastQueryRef.current) {
      sendMessage(lastQueryRef.current);
    }
  }

  function clearMessages() {
    setMessages([]);
  }

  return {
    messages,
    isStreaming,
    sendMessage,
    retryLastMessage,
    clearMessages,
  };
}

async function readSseStream(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const rawEvent of events) {
      const line = rawEvent
        .split("\n")
        .find((entry) => entry.startsWith("data:"));

      if (!line) {
        continue;
      }

      onEvent(JSON.parse(line.replace(/^data:\s*/, "")));
    }
  }
}
