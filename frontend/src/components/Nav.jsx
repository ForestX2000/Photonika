import { useEffect, useState } from "react";
import { BrandDot } from "./Icons";

const LINKS = [
  ["#gallery", "Наборы"],
  ["#studio", "Синтез"],
  ["#how", "Как это работает"],
  ["#numbers", "Результаты"],
];

/** Верхнее меню-«пилюля»: разделы, статус сервера, кнопка запуска, бургер на телефоне. */
export default function Nav({ device }) {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header className={`nav${scrolled ? " scrolled" : ""}`} id="top">
      <a className="brand" href="#top" aria-label="Photonika — на главную">
        <BrandDot />
        <span>Photonika</span>
      </a>
      <nav
        className={`nav-links${open ? " open" : ""}`}
        id="navLinks"
        aria-label="Разделы"
        onClick={(e) => e.target.closest("a") && setOpen(false)}
      >
        {LINKS.map(([href, label]) => (
          <a key={href} href={href}>
            {label}
          </a>
        ))}
      </nav>
      <div className="nav-right">
        <span className={`chip chip-device${device.state ? ` ${device.state}` : ""}`} id="deviceChip" title="Устройство расчёта на сервере">
          <span className="pulse" aria-hidden="true" />
          <span id="deviceText">{device.text}</span>
        </span>
        <a className="btn btn-gold btn-sm" href="#studio">
          Запустить синтез
        </a>
        <button
          className="nav-burger"
          id="navBurger"
          aria-label="Меню"
          aria-expanded={open}
          aria-controls="navLinks"
          onClick={() => setOpen((v) => !v)}
        >
          <span />
          <span />
        </button>
      </div>
    </header>
  );
}
