import { forwardRef } from "react";
import SectionHead from "../SectionHead";
import DoeCard from "./DoeCard";
import MetricsTable from "./MetricsTable";
import Planes from "./Planes";
import Stats from "./Stats";
import Viewer from "./Viewer";

/** Результат синтеза: цифры, ДОЭ, 4 плоскости, 3D-просмотр, таблица метрик. */
const Results = forwardRef(function Results({ result, distances, criticalMm, onError }, ref) {
  return (
    <section className="section results" id="results" ref={ref}>
      <div className="container">
        <SectionHead
          eyebrow={`${result.method_title} · ${result.device.toUpperCase()}`}
          eyebrowId="resultEyebrow"
          title="Четыре изображения — один ДОЭ"
        />
        <Stats result={result} />
        <div className="result-grid">
          {/* key: при новом результате карточка сбрасывает вид и сравнение */}
          <DoeCard key={result.id} result={result} onError={onError} />
          <Planes planes={result.planes} />
        </div>
        <Viewer resultId={result.id} distances={distances} criticalMm={criticalMm} />
        <MetricsTable result={result} />
      </div>
    </section>
  );
});

export default Results;
