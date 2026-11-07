import { API } from "../../lib/api";

/** Компактный список тестовых наборов в студии. */
export default function TestsetList({ testsets, status, selected, onPick }) {
  return (
    <div className="testsets" id="testsets" role="list">
      {status === "error" && <p className="muted small">Тестовые наборы недоступны.</p>}
      {status === "ready" && !testsets.length && (
        <p className="muted small">Тестовых наборов нет: python -m ml.cli make-testsets</p>
      )}
      {testsets.map((s) => (
        <button
          key={s.name}
          type="button"
          className="testset"
          role="listitem"
          aria-pressed={selected === s.name}
          data-name={s.name}
          onClick={() => onPick(s)}
        >
          <div className="thumbs">
            {s.images.map((u) => (
              <img key={u} src={API + u} alt="" loading="lazy" />
            ))}
          </div>
          <span>{s.title}</span>
        </button>
      ))}
    </div>
  );
}
