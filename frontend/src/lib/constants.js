export const DEFAULT_DISTANCES = [24.1, 25.3, 26.5, 27.7];

// PCC по плоскостям для картинок первого экрана (photos_classic, GS-200 + Adam-400; tools/make_web_assets.py)
export const HERO_PCC = [0.59, 0.57, 0.55, 0.54];

export const SET_NOTES = {
  photos_classic: "Классика обработки изображений: самый трудный случай",
  mephi_letters: "Крупные буквы на чёрном фоне — видно спекл",
  text_binary: "Текст, силуэт и фигуры: резкие края",
  digits: "Цифры «2026» — простые бинарные формы",
};
export const FEATURED_ORDER = ["photos_classic", "mephi_letters", "text_binary", "digits"];

export const METHOD_TAGS = {
  "net+gs": ["быстро", "fast"],
  net: ["мгновенно", "fast"],
  "net+adam": ["точнее", ""],
  "gs+adam": ["эталон · долго", "slow"],
  gs: ["классика", ""],
  adam: ["долго", "slow"],
};

// Какие параметры показывать для метода
export const METHOD_PARAMS = {
  "net+gs": ["refine_steps"],
  net: [],
  "net+adam": ["refine_steps"],
  "gs+adam": ["gs_iters", "adam_steps"],
  gs: ["gs_iters"],
  adam: ["adam_steps"],
};

export const PARAM_FIELDS = [
  { key: "gs_iters", id: "gsIters", label: "Итераций GS", min: 10, max: 500, step: 10 },
  { key: "adam_steps", id: "adamSteps", label: "Шагов Adam", min: 50, max: 1000, step: 50 },
  { key: "refine_steps", id: "refineSteps", label: "Шагов доуточнения после сети", min: 0, max: 200, step: 5 },
];
export const DEFAULT_PARAMS = { gs_iters: 200, adam_steps: 400, refine_steps: 25 };

export const Z_MIN = 23;
export const Z_MAX = 29;
