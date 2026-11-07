import { useReveal } from "../lib/hooks";

const NUMBERS = [
  ["0,57", "PCC на фото — потолок даже у долгой оптимизации"],
  ["4×", "больше целевых пикселей, чем степеней свободы ДОЭ"],
  ["0,54", "PCC «трудной» Луны вместо 0,26 у чистого GS"],
  ["8 бит", "метрики только после сохранения в PNG"],
];

function NumberItem({ value, text }) {
  const [ref, reveal] = useReveal();
  return (
    <div className={`number ${reveal}`} ref={ref}>
      <b>{value}</b>
      <span>{text}</span>
    </div>
  );
}

/** «Что показали замеры»: ряд крупных цифр. */
export default function Numbers() {
  const [ref, reveal] = useReveal();
  return (
    <section className="numbers" id="numbers">
      <div className="container numbers-inner">
        <h2 className={`numbers-title ${reveal}`} ref={ref}>
          Что показали
          <br />
          замеры
        </h2>
        {NUMBERS.map(([value, text]) => (
          <NumberItem key={value} value={value} text={text} />
        ))}
      </div>
    </section>
  );
}
