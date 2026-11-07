import { METHOD_PARAMS, PARAM_FIELDS } from "../../lib/constants";

/** «Параметры метода»: ползунки только для параметров выбранного метода. */
export default function MethodParams({ method, params, onChange }) {
  // До загрузки методов показываем все ползунки (как и раньше), потом — только нужные методу
  const visible = method ? METHOD_PARAMS[method] || [] : PARAM_FIELDS.map((f) => f.key);
  return (
    <details className="advanced" id="advanced" hidden={visible.length === 0}>
      <summary>Параметры метода</summary>
      {PARAM_FIELDS.map((f) => (
        <div key={f.key} className={`range-row${visible.includes(f.key) ? "" : " off"}`} data-param={f.key}>
          <label htmlFor={f.id}>
            {f.label} <output id={`${f.id}Out`}>{params[f.key]}</output>
          </label>
          <input
            type="range"
            id={f.id}
            min={f.min}
            max={f.max}
            step={f.step}
            value={params[f.key]}
            onChange={(e) => onChange(f.key, Number(e.target.value))}
          />
        </div>
      ))}
    </details>
  );
}
