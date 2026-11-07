import { BrandDot } from "./Icons";

const SECTIONS = [
  ["#gallery", "Готовые наборы"],
  ["#studio", "Студия синтеза"],
  ["#how", "Как это работает"],
  ["#numbers", "Результаты"],
];

/** Подвал: описание, разделы, кейс, параметры. */
export default function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-grid">
          <div className="footer-brand">
            <a className="brand" href="#top">
              <BrandDot />
              <span>Photonika</span>
            </a>
            <p className="muted">
              Нейросетевой синтез фазовых дифракционных оптических элементов: один ДОЭ — четыре изображения на четырёх
              расстояниях.
            </p>
          </div>
          <div>
            <h4>Разделы</h4>
            {SECTIONS.map(([href, label]) => (
              <a key={href} href={href}>
                {label}
              </a>
            ))}
          </div>
          <div>
            <h4>Кейс</h4>
            <p className="muted">
              Хакатон НИЯУ МИФИ · лаборатория «Фотоника и информационная оптика», каф. № 37 «Лазерная физика».
            </p>
          </div>
          <div>
            <h4>Параметры</h4>
            <p className="muted">
              ДОЭ 128×128 · λ = 532 нм
              <br />
              пиксель ПВМС 10 мкм
              <br />z = 24,1 / 25,3 / 26,5 / 27,7 мм
            </p>
          </div>
        </div>
        <div className="footer-bottom">
          <span className="muted small">© 2026 Photonika</span>
          <a className="muted small" href="#top">
            Наверх ↑
          </a>
        </div>
      </div>
    </footer>
  );
}
