import { API } from "../lib/api";
import { FEATURED_ORDER, SET_NOTES } from "../lib/constants";
import { useReveal } from "../lib/hooks";
import SectionHead from "./SectionHead";

function SetCard({ set, index, onPick }) {
  const [ref, reveal] = useReveal(true);
  return (
    <button type="button" className={`set-card ${reveal}`} ref={ref} aria-label={`${set.title}: открыть в студии`} onClick={() => onPick(set)}>
      <div className="mosaic">
        {set.images.map((u) => (
          <img key={u} src={API + u} alt="" loading="lazy" />
        ))}
      </div>
      <span className="set-num glass">НАБОР №{index + 1}</span>
      <div className="set-meta">
        <h3>{set.title.split(":")[0]}</h3>
        <p>{SET_NOTES[set.name] || set.title}</p>
        <span className="set-go">
          Синтезировать
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
            <path d="M5 12h14M13 6l6 6-6 6" />
          </svg>
        </span>
      </div>
    </button>
  );
}

/** «Готовые наборы»: большие карточки, клик подставляет набор в студию. */
export default function Gallery({ testsets, onPick }) {
  const rank = (name) => FEATURED_ORDER.indexOf(name) + 1 || 99;
  const featured = testsets
    .filter((s) => !s.name.startsWith("random"))
    .sort((a, b) => rank(a.name) - rank(b.name))
    .slice(0, 4);

  return (
    <section className="section gallery-section" id="gallery">
      <div className="container">
        <SectionHead eyebrow="выберите набор — и сразу в студию" title="Готовые наборы" />
        <div className="gallery" id="galleryCards">
          {featured.map((s, i) => (
            <SetCard key={s.name} set={s} index={i} onPick={onPick} />
          ))}
        </div>
        <div className="gallery-nav" aria-hidden="true">
          <span />
          <span />
        </div>
      </div>
    </section>
  );
}
