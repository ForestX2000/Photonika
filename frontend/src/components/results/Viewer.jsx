import { useCallback, useEffect, useRef, useState } from "react";
import { sliceUrl } from "../../lib/api";
import { Z_MAX, Z_MIN } from "../../lib/constants";
import { fmt, zLabel } from "../../lib/format";
import { PlayBadge } from "../Icons";

/**
 * 3D-просмотр: модуль поля на любом z. Одновременно идёт не больше одного запроса среза:
 * новый z ставится в очередь, промежуточные пропускаются. «Пролететь» идёт шагами по 0,1 мм,
 * следующий шаг — только после загрузки предыдущей картинки.
 */
export default function Viewer({ resultId, distances, criticalMm }) {
  const [z, setZ] = useState(distances[0]);
  const [playing, setPlaying] = useState(false);
  const [src, setSrc] = useState(undefined);
  const s = useRef({ z: distances[0], pending: false, queued: null, playing: false, dir: 1, raf: 0 });

  const requestSlice = useCallback(
    (zv) => {
      const st = s.current;
      if (st.pending) {
        st.queued = zv;
        return;
      }
      st.pending = true;
      const img = new Image();
      const done = () => {
        st.pending = false;
        if (st.queued !== null) {
          const q = st.queued;
          st.queued = null;
          requestSlice(q);
        } else if (st.playing) {
          step();
        }
      };
      img.onload = () => {
        setSrc(img.src);
        done();
      };
      img.onerror = done;
      img.src = sliceUrl(resultId, zv);
    },
    // step объявлен ниже и работает только с ref-структурой s, поэтому в зависимостях не нужен
    [resultId],
  );

  const goTo = useCallback(
    (zv) => {
      s.current.z = zv;
      setZ(zv);
      requestSlice(zv);
    },
    [requestSlice],
  );

  function step() {
    const st = s.current;
    let zv = Math.round((st.z + 0.1 * st.dir) * 100) / 100;
    if (zv >= Z_MAX) {
      zv = Z_MAX;
      st.dir = -1;
    }
    if (zv <= Z_MIN) {
      zv = Z_MIN;
      st.dir = 1;
    }
    st.raf = requestAnimationFrame(() => st.playing && goTo(zv));
  }

  const stop = () => {
    s.current.playing = false;
    cancelAnimationFrame(s.current.raf);
    setPlaying(false);
  };

  // Новый результат: стоп, z на первую плоскость, загрузить срез
  useEffect(() => {
    const st = s.current;
    st.playing = false;
    cancelAnimationFrame(st.raf);
    st.pending = false;
    st.queued = null;
    setPlaying(false);
    goTo(distances[0]);
    return () => {
      st.playing = false;
      cancelAnimationFrame(st.raf);
    };
  }, [resultId, distances, goTo]);

  const toggle = () => {
    if (s.current.playing) return stop();
    s.current.playing = true;
    setPlaying(true);
    step();
    return undefined;
  };

  const near = distances.reduce((best, d, i) => (Math.abs(d - z) < Math.abs(distances[best] - z) ? i : best), 0);
  const inFocus = Math.abs(distances[near] - z) < 0.2;
  const crit = criticalMm ?? 24.06;

  return (
    <div className="viewer card spot" id="viewer">
      <div className="viewer-copy">
        <p className="eyebrow">3D-просмотр</p>
        <h3 className="viewer-title">
          Пройдите
          <br />
          вдоль луча
        </h3>
        <p className="muted">
          Двигайте плоскость наблюдения: изображения по очереди фокусируются на своих расстояниях. Это и есть
          «трёхмерная сцена» из одного плоского элемента.
        </p>
        <button className="btn btn-ghost" type="button" id="playBtn" onClick={toggle}>
          <PlayBadge />
          <span id="playLabel">{playing ? "Остановить" : "Пролететь"}</span>
        </button>
      </div>
      <div className="viewer-stage">
        <div className="viewer-img">
          <img id="sliceImg" alt="Модуль поля на выбранном расстоянии" width="128" height="128" src={src} />
          <span className="z-badge" id="zBadge">
            {zLabel(z)}
          </span>
          <span className={`focus-badge${inFocus ? " on" : ""}`} id="focusBadge">
            {inFocus ? `в фокусе: ${near + 1}` : `ближайшая: ${near + 1}`}
          </span>
        </div>
        <div className="z-track">
          <input
            type="range"
            id="zRange"
            min={Z_MIN}
            max={Z_MAX}
            step="0.05"
            value={z}
            aria-label="Расстояние z, мм"
            onChange={(e) => {
              stop();
              goTo(Number(e.target.value));
            }}
          />
          <div className="z-marks" id="zMarks">
            {distances.map((d, i) => (
              <span key={d} style={{ left: `${((d - Z_MIN) / (Z_MAX - Z_MIN)) * 100}%` }}>
                {i + 1}
              </span>
            ))}
          </div>
        </div>
        <p className="muted small" id="zNote">
          {z < crit ? `z < ${fmt(crit, 2)} мм — квадратичная фаза дискретизирована с алиасингом, картинка менее достоверна.` : ""}
        </p>
      </div>
    </div>
  );
}
