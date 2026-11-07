import { useReveal } from "../lib/hooks";
import SectionHead from "./SectionHead";

const PIPELINE = [
  ["01", "4 PNG", "128×128, 8 бит"],
  ["02", "Нейросеть", "фаза за один проход"],
  ["03", "doe.png", "0 = −π, 255 = +π"],
  ["04", "Френель × 4", "24,1 · 25,3 · 26,5 · 27,7 мм"],
  ["05", "PCC · SSIM", "по 8-битным PNG"],
];

const CARDS = [
  {
    accent: true,
    title: "Френель с одним FFT",
    text: "Поле в плоскости восстановления — одно преобразование Фурье от ДОЭ, умноженного на квадратичную фазу своего расстояния. Четыре расстояния — четыре FFT.",
    icon: <path d="M6 24h8m20 0h8M14 12v24M34 12v24M14 18c6 0 6 12 20 12M14 30c6 0 6-12 20-12" />,
  },
  {
    title: "Нейросеть",
    text: "U-Net по четырём изображениям предсказывает фазу поля в каждой плоскости; одно обратное распространение даёт ДОЭ. Обучение без разметки — через дифференцируемую физику.",
    icon: (
      <>
        <circle cx="10" cy="14" r="4" />
        <circle cx="10" cy="34" r="4" />
        <circle cx="24" cy="24" r="4" />
        <circle cx="38" cy="14" r="4" />
        <circle cx="38" cy="34" r="4" />
        <path d="M14 14l6 8m-6 12l6-8m8-2l6-6m-6 10l6 6" />
      </>
    ),
  },
  {
    title: "Метрики жюри",
    text: "PCC и SSIM по 8-битным PNG: ДОЭ сохраняется, перечитывается, восстанавливается. Амплитуды, каждое изображение нормировано на максимум.",
    icon: (
      <>
        <path d="M8 38V10m0 28h32" />
        <path d="M14 30l7-8 6 5 11-13" />
        <circle cx="38" cy="14" r="2.5" />
      </>
    ),
  },
  {
    title: "Итерационный эталон",
    text: "Многоплоскостный Gerchberg–Saxton и Adam по фазе — сотни итераций. С ними сеть сравнивается по качеству и по времени.",
    icon: (
      <>
        <path d="M24 8a16 16 0 1 0 16 16" />
        <path d="M40 8v10H30" />
        <path d="M24 16v8l6 4" />
      </>
    ),
  },
];

function HowCard({ card }) {
  const [ref, reveal] = useReveal();
  return (
    <article className={`how-card${card.accent ? " is-accent" : ""} spot ${reveal}`} ref={ref}>
      <svg className="how-icon" viewBox="0 0 48 48" aria-hidden="true">
        {card.icon}
      </svg>
      <h3>{card.title}</h3>
      <p>{card.text}</p>
    </article>
  );
}

function Formula({ criticalMm }) {
  const [ref, reveal] = useReveal();
  return (
    <div className={`formula card spot ${reveal}`} ref={ref}>
      <div className="formula-math" aria-label="Формула Френеля с одним преобразованием Фурье">
        <math display="block">
          <mi>K</mi>
          <mo>(</mo>
          <mi>x</mi>
          <mo>,</mo>
          <mi>y</mi>
          <mo>)</mo>
          <mo>=</mo>
          <mfrac>
            <msup>
              <mi>e</mi>
              <mrow>
                <mi>j</mi>
                <mi>k</mi>
                <mi>z</mi>
              </mrow>
            </msup>
            <mrow>
              <mi>j</mi>
              <mi>λ</mi>
              <mi>z</mi>
            </mrow>
          </mfrac>
          <msup>
            <mi>e</mi>
            <mrow>
              <mfrac>
                <mrow>
                  <mi>j</mi>
                  <mi>π</mi>
                </mrow>
                <mrow>
                  <mi>λ</mi>
                  <mi>z</mi>
                </mrow>
              </mfrac>
              <mo>(</mo>
              <msup>
                <mi>x</mi>
                <mn>2</mn>
              </msup>
              <mo>+</mo>
              <msup>
                <mi>y</mi>
                <mn>2</mn>
              </msup>
              <mo>)</mo>
            </mrow>
          </msup>
          <mo>·</mo>
          <mi>FFT</mi>
          <mo>[</mo>
          <mi>H</mi>
          <mo>(</mo>
          <mi>u</mi>
          <mo>,</mo>
          <mi>v</mi>
          <mo>)</mo>
          <msup>
            <mi>e</mi>
            <mrow>
              <mfrac>
                <mrow>
                  <mi>j</mi>
                  <mi>π</mi>
                </mrow>
                <mrow>
                  <mi>λ</mi>
                  <mi>z</mi>
                </mrow>
              </mfrac>
              <mo>(</mo>
              <msup>
                <mi>u</mi>
                <mn>2</mn>
              </msup>
              <mo>+</mo>
              <msup>
                <mi>v</mi>
                <mn>2</mn>
              </msup>
              <mo>)</mo>
            </mrow>
          </msup>
          <mo>]</mo>
        </math>
      </div>
      <div className="formula-notes">
        <div>
          <b id="critDist">{criticalMm}</b>
          <span>критическое расстояние N·Δx²/λ — все расстояния чуть больше, алиасинга нет</span>
        </div>
        <div>
          <b>10,0–11,5 мкм</b>
          <span>шаг пикселя восстановления λz/(NΔx)</span>
        </div>
        <div>
          <b>1,2 мм</b>
          <span>между плоскостями — отсюда взаимные помехи изображений</span>
        </div>
      </div>
    </div>
  );
}

/** «Как это работает»: конвейер из 5 шагов, 4 карточки и формула Френеля. */
export default function How({ criticalLabel }) {
  const [pipeRef, pipeReveal] = useReveal();
  return (
    <section className="section" id="how">
      <div className="container">
        <SectionHead eyebrow="физика, нейросеть и честные метрики" title="Как это работает" />

        <ol className={`pipeline spot ${pipeReveal}`} ref={pipeRef}>
          {PIPELINE.map(([num, title, sub]) => (
            <li key={num}>
              <span className="pipe-num">{num}</span>
              <b>{title}</b>
              <small>{sub}</small>
            </li>
          ))}
        </ol>

        <div className="how-grid">
          {CARDS.map((c) => (
            <HowCard key={c.title} card={c} />
          ))}
        </div>

        <Formula criticalMm={criticalLabel} />
      </div>
    </section>
  );
}
