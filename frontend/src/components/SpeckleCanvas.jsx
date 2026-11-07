import { useEffect, useRef } from "react";
import { prefersReducedMotion } from "../lib/hooks";

/** Фон первого экрана: дрейфующий «спекл» — мерцающие точки лазерного цвета. */
export default function SpeckleCanvas() {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas.getContext("2d");
    const reduce = prefersReducedMotion();
    let w = 0;
    let h = 0;
    let dots = [];
    let raf = 0;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const n = Math.round((w * h) / 9000);
      dots = Array.from({ length: n }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        r: Math.random() * 1.4 + 0.2,
        v: Math.random() * 0.25 + 0.05,
        p: Math.random() * Math.PI * 2,
        g: Math.random() < 0.55,
      }));
    };
    const draw = (t) => {
      ctx.clearRect(0, 0, w, h);
      for (const d of dots) {
        const a = 0.25 + 0.75 * Math.abs(Math.sin(d.p + t / 1400));
        ctx.fillStyle = d.g ? `rgba(92,255,143,${a * 0.55})` : `rgba(234,243,237,${a * 0.35})`;
        ctx.beginPath();
        ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
        ctx.fill();
        if (!reduce) {
          d.x += d.v;
          if (d.x > w + 4) d.x = -4;
        }
      }
      if (!reduce) raf = requestAnimationFrame(draw);
    };
    const onResize = () => {
      cancelAnimationFrame(raf);
      resize();
      draw(0);
    };

    resize();
    draw(0);
    window.addEventListener("resize", onResize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  return <canvas className="hero-canvas" id="heroCanvas" aria-hidden="true" ref={ref} />;
}
