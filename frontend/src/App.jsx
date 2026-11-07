import { useCallback, useEffect, useRef, useState } from "react";
import Footer from "./components/Footer";
import Gallery from "./components/Gallery";
import Hero from "./components/Hero";
import How from "./components/How";
import Marquee from "./components/Marquee";
import Nav from "./components/Nav";
import Numbers from "./components/Numbers";
import Results from "./components/results/Results";
import Studio from "./components/studio/Studio";
import Toast from "./components/Toast";
import { api, getInfo, getTestsets, synthesize } from "./lib/api";
import { DEFAULT_DISTANCES } from "./lib/constants";
import { fmt } from "./lib/format";
import { useSpotlight } from "./lib/hooks";
import { useStudio } from "./lib/useStudio";

export default function App() {
  const [info, setInfo] = useState(null);
  const [infoError, setInfoError] = useState(false);
  const [distances, setDistances] = useState(DEFAULT_DISTANCES);
  const [testsets, setTestsets] = useState([]);
  const [testsetStatus, setTestsetStatus] = useState("loading");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [runError, setRunError] = useState(null);
  const [toast, setToast] = useState(null);
  const toastTimer = useRef(0);
  const resultsRef = useRef(null);

  const showToast = useCallback((text, ok = false) => {
    setToast({ text, ok });
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 5200);
  }, []);

  const studio = useStudio({ onTooManyFiles: () => showToast("Взяты первые 4 файла по имени.") });
  const { setMethod } = studio;
  useSpotlight();

  // Данные с сервера: параметры физики и методы, тестовые наборы
  useEffect(() => {
    getInfo()
      .then((data) => {
        setInfo(data);
        setDistances(data.distances_mm);
        setMethod(data.default_method);
      })
      .catch(() => {
        setInfoError(true);
        showToast("Не удалось подключиться к серверу. Запустите: python -m backend");
      });
    getTestsets()
      .then((sets) => {
        setTestsets(sets);
        setTestsetStatus("ready");
      })
      .catch(() => setTestsetStatus("error"));
  }, [setMethod, showToast]);

  // Ошибка под кнопкой видна до следующего изменения входных данных
  useEffect(() => setRunError(null), [studio.slots, studio.method]);

  // Новый результат — плавно прокрутить к нему
  useEffect(() => {
    if (result) resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [result]);

  const ready = studio.slots.every(Boolean) && studio.method && info;
  let hint;
  if (!info) hint = "Ожидание сервера…";
  else if (!studio.slots.every(Boolean)) {
    const n = studio.slots.filter(Boolean).length;
    hint = n ? `Добавлено ${n} из 4 изображений.` : "Добавьте 4 изображения или выберите тестовый набор.";
  } else {
    hint =
      studio.method?.includes("adam") && !studio.method.startsWith("net")
        ? "Итерационный эталон: на CPU это несколько секунд."
        : "Готово к синтезу.";
  }

  const startRun = async () => {
    if (busy) return;
    setBusy(true);
    setRunError(null);
    setElapsed(0);
    const started = performance.now();
    const tick = setInterval(() => setElapsed((performance.now() - started) / 1000), 100);
    let error = null;
    try {
      const fd = new FormData();
      fd.append("method", studio.method);
      fd.append("gs_iters", studio.params.gs_iters);
      fd.append("adam_steps", studio.params.adam_steps);
      fd.append("refine_steps", studio.params.refine_steps);
      if (studio.testset && studio.slots.every((s) => s.testset === studio.testset)) {
        fd.append("testset", studio.testset);
      } else {
        const blobs = await Promise.all(
          studio.slots.map(async (s) => s.file || (await api(s.url.replace(/^https?:\/\/[^/]+/, ""))).blob()),
        );
        blobs.forEach((b, i) => fd.append(`image_${i + 1}`, b, b.name || `image_${i + 1}.png`));
      }
      setResult(await synthesize(fd));
    } catch (e) {
      error = e.message;
      showToast(e.message);
    } finally {
      clearInterval(tick);
      setBusy(false);
      setRunError(error);
    }
  };

  const run = {
    ready: Boolean(ready),
    busy,
    start: startRun,
    label: busy ? `Считаю… ${fmt(elapsed, 1)} с` : "Синтезировать ДОЭ",
    hint: runError || hint,
    error: Boolean(runError),
  };

  const device = info
    ? {
        state: "ok",
        text: (info.gpu ? `GPU · ${info.gpu.replace("NVIDIA GeForce ", "")}` : "CPU") + (info.weights ? " · сеть готова" : ""),
      }
    : infoError
      ? { state: "err", text: "сервер недоступен" }
      : { state: "", text: "подключение…" };

  const pickFromGallery = (set) => {
    studio.chooseTestset(set);
    document.getElementById("studio")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <>
      <Nav device={device} />
      <Hero />
      <Marquee />
      <Gallery testsets={testsets} onPick={pickFromGallery} />
      <Studio
        studio={studio}
        distances={distances}
        testsets={testsets}
        testsetStatus={testsetStatus}
        methods={info ? info.methods : infoError ? "error" : null}
        run={run}
      />
      {result && (
        <Results
          ref={resultsRef}
          result={result}
          distances={distances}
          criticalMm={info?.critical_distance_mm}
          onError={showToast}
        />
      )}
      <How criticalLabel={info ? `${fmt(info.critical_distance_mm, 2)} мм` : "24,06 мм"} />
      <Numbers />
      <Footer />
      <Toast toast={toast} />
    </>
  );
}
