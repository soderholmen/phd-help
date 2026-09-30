// The corpus client against a fake fetch: URLs, methods, query params,
// and the 409-with-error-body path.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "./corpus";

const fetchMock = vi.fn();

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => {
  vi.clearAllMocks();
});

function respond(body: unknown, ok = true, status = 200) {
  fetchMock.mockResolvedValue({
    ok,
    status,
    json: async () => body,
  });
}

describe("corpus client", () => {
  it("lists docs", async () => {
    respond([{ doc_id: "d1" }]);
    await expect(api.listDocs()).resolves.toEqual([{ doc_id: "d1" }]);
    expect(fetchMock).toHaveBeenCalledWith("/corpus/docs");
  });

  it("uploads raw bytes with only the filled meta params", async () => {
    respond({ doc_id: "d1", status: "queued", new: true });
    const bytes = new ArrayBuffer(4);
    await api.uploadPdf(bytes, { title: "Attention", arxiv: "" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/corpus/upload?title=Attention");
    expect(init.method).toBe("POST");
    expect(init.body).toBe(bytes);
  });

  it("uploads with no query when meta is empty", async () => {
    respond({ doc_id: "d1", status: "queued", new: true });
    await api.uploadPdf(new ArrayBuffer(4));
    expect(fetchMock.mock.calls[0][0]).toBe("/corpus/upload");
  });

  it("retry/pin/unpin hit their doors", async () => {
    respond({ doc_id: "d1", status: "queued" });
    await api.retry("d1");
    expect(fetchMock).toHaveBeenLastCalledWith("/corpus/d1/retry", { method: "POST" });
    respond({ doc_id: "d1", pinned_in: ["my-paper"] });
    await api.pin("d1");
    expect(fetchMock).toHaveBeenLastCalledWith("/corpus/d1/pin", { method: "POST" });
    await api.unpin("d1");
    expect(fetchMock).toHaveBeenLastCalledWith("/corpus/d1/unpin", { method: "POST" });
  });

  it("surfaces the server's error text on 409", async () => {
    respond({ error: "no such doc" }, false, 409);
    await expect(api.retry("ghost")).rejects.toThrow("no such doc");
  });

  it("falls back to the status when the body has no error", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 500, json: async () => { throw new Error("nope"); } });
    await expect(api.listDocs()).rejects.toThrow("HTTP 500");
  });
});
