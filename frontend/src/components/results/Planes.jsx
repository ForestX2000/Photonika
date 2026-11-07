import { useState } from "react";
import { fmt, zLabel } from "../../lib/format";

const VIEWS = [
  ["recon", "Восстановление"],
  ["laser", "Лазер"],
  ["source", "Исходное"],
];

/** Четыре карточки плоскостей; при наведении видно исходное изображение. */
export default function Planes({ planes }) {
  const [view, setView] = useState("recon");
  return (
    <div className="planes-wrap">
      <div className="planes-head">
        <div className="seg" role="tablist" aria-label="Что показывать">
          {VIEWS.map(([key, label]) => (
            <button key={key} type="button" role="tab" aria-selected={view === key} data-view={key} onClick={() => setView(key)}>
              {label}
            </button>
          ))}
        </div>
        <span className="muted small">наведите на карточку, чтобы увидеть исходное</span>
      </div>
      <div className="planes" id="planes">
        {planes.map((p) => {
          const main = view === "source" ? p.source : view === "laser" ? p.recon_laser : p.recon;
          const alt = view === "source" ? p.recon : p.source;
          return (
            <article className="plane-card" key={p.index}>
              <img alt={`Плоскость ${p.index}: ${view === "source" ? "исходное" : "восстановление"}`} src={main} />
              <img className="alt" alt="" src={alt} />
              <div className="plane-meta">
                <h4>Плоскость №{p.index}</h4>
                <p>z = {zLabel(p.distance_mm)}</p>
                <div className="plane-chips">
                  <span className="chip">
                    PCC <b>{fmt(p.pcc)}</b>
                  </span>
                  <span className="chip">
                    SSIM <b>{fmt(p.ssim)}</b>
                  </span>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}
