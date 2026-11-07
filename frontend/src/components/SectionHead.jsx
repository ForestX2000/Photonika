import { useReveal } from "../lib/hooks";

/** Заголовок секции: надзаголовок капсом + крупный заголовок, с появлением при прокрутке. */
export default function SectionHead({ eyebrow, title, eyebrowId }) {
  const [ref, reveal] = useReveal();
  return (
    <div className={`section-head ${reveal}`} ref={ref}>
      <p className="eyebrow center" id={eyebrowId}>
        {eyebrow}
      </p>
      <h2 className="section-title">{title}</h2>
    </div>
  );
}
