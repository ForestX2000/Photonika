import { useRef } from "react";
import { METHOD_TAGS } from "../../lib/constants";

/** Выбор метода синтеза: радиогруппа с навигацией стрелками. Недоступные методы — в конце. */
export default function MethodPicker({ methods, selected, onSelect }) {
  const box = useRef(null);
  if (methods === null) return <div className="methods" id="methods" role="radiogroup" aria-label="Метод синтеза" />;
  if (methods === "error") {
    return (
      <div className="methods" id="methods" role="radiogroup" aria-label="Метод синтеза">
        <p className="muted small">Методы появятся после подключения к серверу.</p>
      </div>
    );
  }

  const sorted = [...methods].sort((a, b) => Number(b.available) - Number(a.available));
  const available = sorted.filter((m) => m.available);
  const move = (id, step) => {
    const k = available.findIndex((m) => m.id === id);
    const next = available[(k + step + available.length) % available.length];
    onSelect(next.id);
    box.current.querySelector(`[data-id="${CSS.escape(next.id)}"]`)?.focus();
  };

  return (
    <div className="methods" id="methods" role="radiogroup" aria-label="Метод синтеза" ref={box}>
      {sorted.map((m) => {
        const [tag, cls] = m.available ? METHOD_TAGS[m.id] || ["", ""] : ["нужны веса", ""];
        const on = m.id === selected;
        return (
          <div
            key={m.id}
            className="method"
            role="radio"
            tabIndex={on ? 0 : -1}
            data-id={m.id}
            aria-checked={on}
            aria-disabled={m.available ? undefined : "true"}
            onClick={() => m.available && onSelect(m.id)}
            onKeyDown={(e) => {
              if (["ArrowDown", "ArrowRight"].includes(e.key)) {
                e.preventDefault();
                move(m.id, 1);
              }
              if (["ArrowUp", "ArrowLeft"].includes(e.key)) {
                e.preventDefault();
                move(m.id, -1);
              }
              if (e.key === " " || e.key === "Enter") {
                e.preventDefault();
                if (m.available) onSelect(m.id);
              }
            }}
          >
            <span className="radio" aria-hidden="true" />
            <span>
              <b>{m.title}</b>
              <small>{m.description}</small>
            </span>
            {tag && <span className={`tag ${cls}`}>{tag}</span>}
          </div>
        );
      })}
    </div>
  );
}
