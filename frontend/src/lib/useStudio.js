import { useCallback, useEffect, useRef, useState } from "react";
import { API } from "./api";
import { DEFAULT_PARAMS } from "./constants";

const EMPTY = [null, null, null, null];

/**
 * Состояние студии: 4 ячейки ({url, file?} или {url, testset, index}), выбранный тестовый набор,
 * метод и его параметры. Для загруженных файлов создаются object URL — они освобождаются,
 * когда файл убирают или заменяют.
 */
export function useStudio({ onTooManyFiles } = {}) {
  const [slots, setSlotsState] = useState(EMPTY);
  const [testset, setTestset] = useState(null);
  const [method, setMethod] = useState(null);
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const current = useRef(EMPTY);

  const commit = (next) => {
    current.current = next;
    setSlotsState(next);
  };
  const revoke = (slot) => slot?.file && URL.revokeObjectURL(slot.url);
  const fileSlot = (file) => ({ file, url: URL.createObjectURL(file) });

  const setFile = useCallback((i, file) => {
    const next = [...current.current];
    revoke(next[i]);
    next[i] = fileSlot(file);
    commit(next);
    setTestset(null);
  }, []);

  const removeSlot = useCallback((i) => {
    const next = [...current.current];
    revoke(next[i]);
    next[i] = null;
    commit(next);
    setTestset(null);
  }, []);

  const distribute = useCallback(
    (files, start = 0) => {
      const sorted = [...files].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
      const next = [...current.current];
      sorted.slice(0, 4).forEach((f, k) => {
        const i = sorted.length >= 4 ? k : (start + k) % 4;
        revoke(next[i]);
        next[i] = fileSlot(f);
      });
      commit(next);
      setTestset(null);
      if (sorted.length > 4) onTooManyFiles?.();
    },
    [onTooManyFiles],
  );

  const chooseTestset = useCallback((set) => {
    current.current.forEach(revoke);
    commit(set.images.map((u, k) => ({ url: API + u, testset: set.name, index: k + 1 })));
    setTestset(set.name);
  }, []);

  const clear = useCallback(() => {
    current.current.forEach(revoke);
    commit(EMPTY);
    setTestset(null);
  }, []);

  const setParam = useCallback((key, value) => setParams((p) => ({ ...p, [key]: value })), []);

  // освободить object URL при закрытии страницы/размонтировании
  useEffect(() => () => current.current.forEach(revoke), []);

  return { slots, testset, method, params, setFile, removeSlot, distribute, chooseTestset, clear, setMethod, setParam };
}
