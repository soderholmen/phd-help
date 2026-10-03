// The doors' shared error discipline: non-2xx throws with the server's
// error text when it has one (the doors answer 400/404/409 with
// {"error": ...}). json() uses it; the byte doors (download) call it
// before touching the body as a blob.
export async function throwIfError(r: Response): Promise<void> {
  if (r.ok) return;
  let detail = "";
  try {
    detail = (await r.json())?.error ?? "";
  } catch {
    /* body wasn't JSON */
  }
  throw new Error(detail || `HTTP ${r.status}`);
}

export async function json<T>(r: Response): Promise<T> {
  await throwIfError(r);
  return (await r.json()) as T;
}
