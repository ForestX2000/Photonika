import { useEffect, useState } from "react";
import { DEFAULT_DISTANCES, HERO_PCC } from "../lib/constants";
import { fmt, zLabel } from "../lib/format";
import { prefersReducedMotion } from "../lib/hooks";
import { ArrowRight, PlayBadge } from "./Icons";
import SpeckleCanvas from "./SpeckleCanvas";

const PLANES = [1, 2, 3, 4];
const FACTS = [
  { href: "#how", body: <>Фазовый ДОЭ <b>128×128</b>, фаза от −π до π, сохраняется как 8-битный PNG.</> },
  { href: "#how", body: <>Дифракция Френеля с <b>одним FFT</b>, λ = 532 нм, пиксель ПВМС 10 мкм.</> },
  { href: "#numbers", body: <>Качество — <b>PCC и SSIM</b> по амплитудам, каждое изображение нормировано на 1.</> },
];

/** Первый экран: заголовок, 3D-сцена «ДОЭ → 4 плоскости», нумерация 01–04, факты. */
export default function Hero() {
  const [current, setCurrent] = useState(0);
  const [cycle, setCycle] = useState(0); // перезапуск автопереключения после клика

  useEffect(() => {
    if (prefersReducedMotion()) return undefined;
    const timer = setInterval(() => setCurrent((c) => (c + 1) % 4), 2600);
    return () => clearInterval(timer);
  }, [cycle]);

  const choose = (i) => {
    setCurrent(i);
    setCycle((c) => c + 1);
  };

  return (
    <section className="hero">
      <SpeckleCanvas />
      <div className="hero-glow" aria-hidden="true" />

      <div className="hero-inner container">
        <div className="hero-copy">
          <p className="eyebrow">НИЯУ МИФИ · лаборатория «Фотоника и информационная оптика»</p>
          <h1 className="hero-title">
            Свет
            <br />в{" "}четырёх
            <br />
            <span className="laser-text">плоскостях</span>
          </h1>
          <p className="hero-lead">
            Один фазовый ДОЭ 128×128 — и на расстояниях 24,1, 25,3, 26,5 и 27,7{" "}мм по очереди фокусируются
            четыре разных изображения. Нейросеть рассчитывает такой элемент за один проход.
          </p>
          <div className="hero-actions">
            <a className="btn btn-gold" href="#studio">
              Запустить синтез
              <ArrowRight />
            </a>
            <a className="btn btn-ghost" href="#how">
              <PlayBadge />
              Как это работает
            </a>
          </div>
        </div>

        <div className="scene" id="scene" aria-label="Схема: ДОЭ и четыре плоскости восстановления">
          <div className="beam" aria-hidden="true" />
          <figure className="plate plate-doe">
            <img src="/images/doe_phase.png" alt="Фаза синтезированного ДОЭ" width="128" height="128" />
            <figcaption>ДОЭ · фаза</figcaption>
          </figure>
          {PLANES.map((n, i) => (
            <figure key={n} className={`plate${current === i ? " active" : ""}`} data-plane={n}>
              <img src={`/images/plane_${n}.png`} alt={`Плоскость ${n}`} width="128" height="128" />
              <figcaption>{zLabel(DEFAULT_DISTANCES[i])}</figcaption>
            </figure>
          ))}

          <div className="focus-card glass" aria-live="polite">
            <span className="focus-dot" aria-hidden="true" />
            <div>
              <small>сейчас в фокусе</small>
              <b id="focusTitle">Плоскость 0{current + 1}</b>
              <span id="focusMeta">
                z = {zLabel(DEFAULT_DISTANCES[current])} · PCC {fmt(HERO_PCC[current], 2)}
              </span>
            </div>
            <div className="focus-bar" aria-hidden="true">
              {/* key перезапускает CSS-анимацию заполнения при смене плоскости */}
              <span key={`${current}-${cycle}`} id="focusBar" className="run" />
            </div>
          </div>
          <div className="hero-badge glass" aria-hidden="true">
            <b>1</b>
            <span>ДОЭ</span>
            <i />
            <b>4</b>
            <span>изображения</span>
          </div>
        </div>

        <ol className="plane-index" id="planeIndex" aria-label="Плоскости восстановления">
          {PLANES.map((n, i) => (
            <li key={n} data-plane={n} className={current === i ? "active" : undefined}>
              <button type="button" onClick={() => choose(i)}>
                0{n}
              </button>
            </li>
          ))}
        </ol>
      </div>

      <div className="hero-facts container">
        {FACTS.map((f, i) => (
          <div className="fact" key={i}>
            <p>{f.body}</p>
            <a href={f.href}>
              Подробнее <span aria-hidden="true">→</span>
            </a>
          </div>
        ))}
        <div className="fact-progress" aria-hidden="true">
          <span id="factProgress" style={{ transform: `translateX(${current * 100}%)` }} />
        </div>
      </div>
    </section>
  );
}
