import { useEffect, useRef, useState } from "react";

export const prefersReducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * Появление при прокрутке: возвращает [ref, className]. Элемент получает класс `reveal`,
 * а когда попадает в зону видимости — ещё и `in` (анимация в components.css).
 * immediate — показать сразу после монтирования (для карточек, которые появляются по данным).
 */
export function useReveal(immediate = false) {
  const ref = useRef(null);
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    if (immediate) {
      const raf = requestAnimationFrame(() => setShown(true));
      return () => cancelAnimationFrame(raf);
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setShown(true);
          io.disconnect();
        }
      },
      { threshold: 0.12 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [immediate]);
  return [ref, shown ? "reveal in" : "reveal"];
}

/** Подсветка под курсором: координаты в CSS-переменные --mx/--my у ближайшего элемента .spot. */
export function useSpotlight() {
  useEffect(() => {
    const onMove = (e) => {
      const el = e.target.closest?.(".spot");
      if (!el) return;
      const r = el.getBoundingClientRect();
      el.style.setProperty("--mx", `${e.clientX - r.left}px`);
      el.style.setProperty("--my", `${e.clientY - r.top}px`);
    };
    document.addEventListener("pointermove", onMove, { passive: true });
    return () => document.removeEventListener("pointermove", onMove);
  }, []);
}
