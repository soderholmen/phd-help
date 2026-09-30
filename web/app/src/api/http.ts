// Shared JSON fetch helper: non-2xx throws with the server's error text
// when it has one (the corpus doors answer 409 with {"error": ...}).
export async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = "";
    try {
      detail = (await r.json())?.error ?? "";
    } catch {
      /* body wasn't JSON */
    }
    throw new Error(detail || `HTTP ${r.status}`);
  }
  return (await r.json()) as T;
}
