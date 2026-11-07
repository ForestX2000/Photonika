export const ArrowRight = (props) => (
  <svg viewBox="0 0 24 24" aria-hidden="true" {...props}>
    <path d="M5 12h14M13 6l6 6-6 6" />
  </svg>
);

export const Download = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M12 4v11m0 0l-5-5m5 5l5-5M5 20h14" />
  </svg>
);

export const PlayBadge = () => (
  <span className="play" aria-hidden="true">
    <svg viewBox="0 0 24 24">
      <path d="M8 5v14l11-7z" />
    </svg>
  </span>
);

export const BrandDot = () => <span className="brand-dot" aria-hidden="true" />;
