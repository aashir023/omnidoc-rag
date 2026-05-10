import { AlertCircle, CheckCircle2, Info, X } from "lucide-react";

const styles = {
  success: "border-l-success bg-success/5 text-primary",
  error: "border-l-error bg-error/5 text-primary",
  info: "border-l-accent bg-accent/5 text-primary",
};

const icons = {
  success: <CheckCircle2 size={18} className="text-success" />,
  error: <AlertCircle size={18} className="text-error" />,
  info: <Info size={18} className="text-accent" />,
};

export default function Toast({ toasts }) {
  return (
    <div className="fixed right-6 top-6 z-[100] flex w-full max-w-[380px] flex-col gap-3">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`group relative animate-slide-in overflow-hidden rounded-xl border border-border border-l-4 bg-surface/90 p-4 shadow-2xl backdrop-blur-xl ${styles[toast.type] || styles.info}`}
        >
          <div className="flex items-start gap-3">
            <div className="mt-0.5 shrink-0">{icons[toast.type] || icons.info}</div>
            <div className="flex-1">
              <p className="text-sm font-medium leading-relaxed">
                {toast.message}
              </p>
            </div>
          </div>
          
          <div className="absolute bottom-0 left-0 h-0.5 w-full bg-border-subtle">
            <div
              className={`h-full transition-all duration-[4000ms] linear ${
                toast.type === "success" ? "bg-success" : toast.type === "error" ? "bg-error" : "bg-accent"
              }`}
              style={{ width: "0%", transitionProperty: "width", transitionTimingFunction: "linear" }}
              ref={(el) => {
                if (el) {
                  // Small hack to trigger the transition after mount
                  requestAnimationFrame(() => {
                    el.style.width = "100%";
                  });
                }
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
