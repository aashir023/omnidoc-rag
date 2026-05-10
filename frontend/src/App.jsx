import { useState } from "react";
import { Menu } from "lucide-react";
import ChatArea from "./components/ChatArea.jsx";
import Sidebar from "./components/Sidebar.jsx";
import Toast from "./components/Toast.jsx";
import useChat from "./hooks/useChat.js";
import useDocuments from "./hooks/useDocuments.js";

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [toasts, setToasts] = useState([]);
  const documents = useDocuments(addToast);
  const chat = useChat(documents.activeFiles, addToast);

  function addToast(type, message) {
    const id = crypto.randomUUID();
    setToasts((current) => [...current, { id, type, message }]);
    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id));
    }, 4000);
  }

  const closeSidebar = () => setSidebarOpen(false);

  return (
    <div className="min-h-screen bg-background font-sans text-primary selection:bg-accent/30">
      {/* Mobile Menu Trigger */}
      <button
        type="button"
        aria-label="Open navigation"
        onClick={() => setSidebarOpen(true)}
        className="fixed left-4 top-4 z-40 flex h-10 w-10 items-center justify-center rounded-xl border border-border bg-surface/80 text-secondary shadow-lg backdrop-blur-md transition-all hover:bg-elevated hover:text-primary active:scale-95 md:hidden"
      >
        <Menu size={20} />
      </button>

      {/* Mobile Sidebar Backdrop */}
      {sidebarOpen && (
        <button
          type="button"
          aria-label="Close navigation backdrop"
          onClick={closeSidebar}
          className="fixed inset-0 z-40 bg-background/60 backdrop-blur-sm transition-opacity md:hidden"
        />
      )}

      <Sidebar
        documents={documents}
        onClearConversation={chat.clearMessages}
        isOpen={sidebarOpen}
        onClose={closeSidebar}
      />
      
      <ChatArea chat={chat} activeCount={documents.activeFiles.length} />
      
      <Toast toasts={toasts} />
    </div>
  );
}
