// Обращения к бэкенду (backend/main.py). По умолчанию — тот же адрес, что и у сайта;
// для отдельного сервера задайте VITE_API_URL, например http://127.0.0.1:8000.
export const API = import.meta.env.VITE_API_URL || "";

export async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    let msg = `Ошибка ${res.status}`;
    try {
      const j = await res.json();
      if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* ответ без JSON */
    }
    throw new Error(msg);
  }
  return res;
}

export const getInfo = async () => (await api("/api/info")).json();
export const getTestsets = async () => (await api("/api/testsets")).json();
export const synthesize = async (formData) => (await api("/api/synthesize", { method: "POST", body: formData })).json();
export const compareWithGs = async (id, gsIters = 50) =>
  (await api(`/api/results/${id}/compare?gs_iters=${gsIters}`, { method: "POST" })).json();
export const sliceUrl = (id, z) => `${API}/api/results/${id}/slice.png?z_mm=${z.toFixed(2)}`;
export const downloadUrl = (id) => `${API}/api/results/${id}/download.zip`;
