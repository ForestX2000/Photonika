import { useState } from "react";
import { compareWithGs, downloadUrl } from "../../lib/api";
import { fmt, fmtMs } from "../../lib/format";
import { Download } from "../Icons";

function Compare({ result, data }) {
  const tMax = Math.max(result.time_ms, data.time_ms);
  const faster = data.time_ms / Math.max(result.time_ms, 1e-6);
  const dPcc = result.mean.pcc - data.mean.pcc;
  const rows = [
    ["время, наш", "", (result.time_ms / tMax) * 100, fmtMs(result.time_ms)],
    ["время, GS-50", " ref", (data.time_ms / tMax) * 100, fmtMs(data.time_ms)],
    ["PCC, наш", "", Math.max(0, result.mean.pcc) * 100, fmt(result.mean.pcc)],
    ["PCC, GS-50", " ref", Math.max(0, data.mean.pcc) * 100, fmt(data.mean.pcc)],
  ];
  return (
    <div className="compare" id="compare">
      <h4>{result.method_title} против GS-50</h4>
      {rows.map(([label, ref, width, value]) => (
        <div className="bar-row" key={label}>
          <span>{label}</span>
          <div className={`bar${ref}`}>
            <span style={{ width: `${width}%` }} />
          </div>
          <b>{value}</b>
        </div>
      ))}
      <p className="verdict muted">
        {faster >= 1 ? `В ${fmt(faster, 1)}× быстрее` : `В ${fmt(1 / faster, 1)}× медленнее`} GS-50, PCC{" "}
        {dPcc >= 0 ? "выше" : "ниже"} на {fmt(Math.abs(dPcc))}.
      </p>
    </div>
  );
}

/** Карточка ДОЭ: фаза (цвет / doe.png), скачивание архива, сравнение с GS-50. */
export default function DoeCard({ result, onError }) {
  const [view, setView] = useState("color");
  const [compare, setCompare] = useState(null);
  const [busy, setBusy] = useState(false);

  const runCompare = async () => {
    setBusy(true);
    try {
      setCompare(await compareWithGs(result.id, 50));
    } catch (e) {
      onError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const compareLabel = busy ? "Считаю GS-50…" : compare ? "Пересчитать GS-50" : "Сравнить с GS-50";
  return (
    <article className="card doe-card spot">
      <div className="doe-head">
        <h3>ДОЭ</h3>
        <div className="seg seg-sm" role="tablist" aria-label="Вид ДОЭ">
          {[
            ["color", "цвет"],
            ["raw", "doe.png"],
          ].map(([kind, label]) => (
            <button key={kind} type="button" role="tab" aria-selected={view === kind} data-doe={kind} onClick={() => setView(kind)}>
              {label}
            </button>
          ))}
        </div>
      </div>
      <div className="doe-img">
        <img id="doeImg" alt="Синтезированный ДОЭ" width="128" height="128" src={view === "color" ? result.doe_color : result.doe} />
      </div>
      <p className="muted small" id="doeCaption">
        {view === "color"
          ? "Фаза от −π до π в циклической палитре (−π и +π — один цвет)."
          : "Так выглядит doe.png: 8 бит, 0 = −π, 255 = +π."}
      </p>
      <div className="doe-actions">
        <a className="btn btn-gold btn-block" id="downloadBtn" href={downloadUrl(result.id)} download={`photonika_${result.id}.zip`}>
          <Download />
          Скачать результат (.zip)
        </a>
        <button className="btn btn-ghost btn-block" type="button" id="compareBtn" disabled={busy} onClick={runCompare}>
          {compareLabel}
        </button>
      </div>
      {compare && <Compare result={result} data={compare} />}
    </article>
  );
}
