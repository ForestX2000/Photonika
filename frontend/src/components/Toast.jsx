/** Всплывающее уведомление внизу экрана. */
export default function Toast({ toast }) {
  return (
    <div className={`toast${toast?.ok ? " ok" : ""}`} id="toast" role="status" aria-live="polite" hidden={!toast}>
      {toast?.text}
    </div>
  );
}
