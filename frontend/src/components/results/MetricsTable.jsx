import { fmt } from "../../lib/format";

/** Таблица метрик по плоскостям + строка среднего. */
export default function MetricsTable({ result }) {
  return (
    <div className="card table-card spot">
      <h3>Метрики по плоскостям</h3>
      <p className="muted small">
        Протокол организаторов: сравниваются амплитуды, каждое изображение нормировано на 1. SSIM — локальный
        (skimage, окно 7×7), SSIM со слайда — формула ИСС по всему изображению.
      </p>
      <div className="table-scroll">
        <table className="metrics" id="metricsTable">
          <thead>
            <tr>
              <th>Плоскость</th>
              <th>z, мм</th>
              <th>PCC</th>
              <th>SSIM</th>
              <th>SSIM (слайд)</th>
            </tr>
          </thead>
          <tbody>
            {result.planes.map((p) => (
              <tr key={p.index}>
                <td>{p.index}</td>
                <td>{fmt(p.distance_mm, 1)}</td>
                <td>{fmt(p.pcc, 4)}</td>
                <td>{fmt(p.ssim, 4)}</td>
                <td>{fmt(p.ssim_global, 4)}</td>
              </tr>
            ))}
            <tr>
              <td>среднее</td>
              <td />
              <td>{fmt(result.mean.pcc, 4)}</td>
              <td>{fmt(result.mean.ssim, 4)}</td>
              <td>{fmt(result.mean.ssim_global, 4)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
