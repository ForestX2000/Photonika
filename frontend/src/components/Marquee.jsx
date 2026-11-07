const ITEMS = [
  "Фазовый ДОЭ 128×128",
  "Френель с одним FFT",
  "λ = 532 нм",
  "4 плоскости · 24,1–27,7 мм",
  "U-Net без разметки",
  "PCC · SSIM",
  "Gerchberg–Saxton",
  "8-битный PNG",
];

/** Золотая бегущая лента. Две копии содержимого — бесшовная прокрутка на −50 %. */
export default function Marquee() {
  return (
    <div className="marquee" aria-hidden="true">
      <div className="marquee-track" id="marqueeTrack">
        {[0, 1].map((copy) =>
          ITEMS.map((text) => [<span key={`${copy}-${text}`}>{text}</span>, <i key={`${copy}-${text}-i`}>✦</i>]),
        )}
      </div>
    </div>
  );
}
