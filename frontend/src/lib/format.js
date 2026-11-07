export const fmt = (v, d = 3) =>
  Number(v).toLocaleString("ru-RU", { minimumFractionDigits: d, maximumFractionDigits: d });

export const fmtMs = (ms) =>
  ms >= 1000 ? `${fmt(ms / 1000, ms >= 10000 ? 1 : 2)} с` : `${fmt(ms, ms >= 100 ? 0 : 1)} мс`;

export const zLabel = (z) => `${fmt(z, 1)} мм`;
