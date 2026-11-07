import { useReveal } from "../../lib/hooks";
import SectionHead from "../SectionHead";
import MethodParams from "./MethodParams";
import MethodPicker from "./MethodPicker";
import Slots from "./Slots";
import TestsetList from "./TestsetList";

/** «Студия синтеза»: шаг 1 — изображения, шаг 2 — метод и запуск. */
export default function Studio({ studio, distances, testsets, testsetStatus, methods, run }) {
  const [ref, reveal] = useReveal();
  return (
    <section className="section" id="studio">
      <div className="container">
        <SectionHead eyebrow="загрузите 4 изображения — получите один ДОЭ" title="Студия синтеза" />

        <div className={`studio card spot ${reveal}`} ref={ref}>
          <div className="studio-inputs">
            <div className="step-head">
              <span className="step-num">1</span>
              <div>
                <h3>Изображения</h3>
                <p className="muted">
                  i-е изображение фокусируется на i-м расстоянии. Перетащите файлы или нажмите на ячейку. Цветные
                  приводятся к яркости, другой размер — к 128×128.
                </p>
              </div>
            </div>
            <Slots
              slots={studio.slots}
              distances={distances}
              onSetFile={studio.setFile}
              onDistribute={studio.distribute}
              onRemove={studio.removeSlot}
              onClear={studio.clear}
            />

            <div className="testsets-head">
              <h4>или тестовый набор</h4>
              <span className="muted small">в обучение не входили</span>
            </div>
            <TestsetList testsets={testsets} status={testsetStatus} selected={studio.testset} onPick={studio.chooseTestset} />
          </div>

          <div className="studio-settings">
            <div className="step-head">
              <span className="step-num">2</span>
              <div>
                <h3>Метод</h3>
                <p className="muted">Нейросеть даёт ДОЭ за один проход; итерационные методы — эталон для сравнения.</p>
              </div>
            </div>
            <MethodPicker methods={methods} selected={studio.method} onSelect={studio.setMethod} />
            <MethodParams method={studio.method} params={studio.params} onChange={studio.setParam} />

            <button
              className={`btn btn-gold btn-lg btn-block${run.busy ? " loading" : ""}`}
              type="button"
              id="runBtn"
              disabled={!run.ready || run.busy}
              onClick={run.start}
            >
              <span className="btn-label">{run.label}</span>
              <span className="spinner" aria-hidden="true" />
            </button>
            <p className={`run-hint muted small${run.error ? " err" : ""}`} id="runHint">
              {run.hint}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
