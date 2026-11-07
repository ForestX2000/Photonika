import { fmt, fmtMs } from "../../lib/format";

/** Крупные цифры результата: PCC, SSIM, SSIM по слайду, время. */
export default function Stats({ result }) {
  return (
    <div className="stats card spot" id="stats">
      <div className="stat">
        <b>{fmt(result.mean.pcc)}</b>
        <span>PCC, среднее по 4 плоскостям</span>
      </div>
      <div className="stat">
        <b>{fmt(result.mean.ssim)}</b>
        <span>SSIM (skimage), среднее</span>
      </div>
      <div className="stat">
        <b>{fmt(result.mean.ssim_global)}</b>
        <span>SSIM по формуле слайда</span>
      </div>
      <div className="stat">
        <b>{fmtMs(result.time_ms)}</b>
        <span>
          время синтеза · <span className="sub">{result.method_title}</span>
        </span>
      </div>
    </div>
  );
}
