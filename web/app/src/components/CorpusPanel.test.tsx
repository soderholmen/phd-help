import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { CorpusPanel } from "./CorpusPanel";
import type { CorpusDoc } from "../types";

const doc = (over: Partial<CorpusDoc>): CorpusDoc => ({
  doc_id: "d1",
  title: "Attention Is All You Need",
  status: "indexed",
  error: null,
  arxiv: "1706.03762",
  doi: "",
  year: "2017",
  source: "upload",
  pinned_in: [],
  pinned_here: false,
  chunk_count: 29,
  ...over,
});

const noop = {
  onUpload: async () => {},
  onRetry: () => {},
  onPin: () => {},
  onUnpin: () => {},
};

describe("CorpusPanel", () => {
  it("shows status, title and chunk count", () => {
    render(<CorpusPanel docs={[doc({})]} {...noop} />);
    expect(screen.getByText("indexed")).toBeInTheDocument();
    expect(screen.getByText(/Attention Is All You Need/)).toBeInTheDocument();
    expect(screen.getByText("29 chunks")).toBeInTheDocument();
  });

  it("a failed doc shows its error and a retry that names the doc", () => {
    const onRetry = vi.fn();
    render(
      <CorpusPanel docs={[doc({ status: "failed", error: "MinerU died" })]} {...noop} onRetry={onRetry} />,
    );
    expect(screen.getByText("MinerU died")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalledWith("d1");
  });

  it("pin and unpin follow pinned_here", () => {
    const onPin = vi.fn();
    const onUnpin = vi.fn();
    const { rerender } = render(
      <CorpusPanel docs={[doc({})]} {...noop} onPin={onPin} onUnpin={onUnpin} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Pin" }));
    expect(onPin).toHaveBeenCalledWith("d1");
    rerender(
      <CorpusPanel docs={[doc({ pinned_here: true })]} {...noop} onPin={onPin} onUnpin={onUnpin} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Unpin" }));
    expect(onUnpin).toHaveBeenCalledWith("d1");
  });

  it("uploads the picked file's bytes with the meta fields", async () => {
    const onUpload = vi.fn();
    render(<CorpusPanel docs={[]} {...noop} onUpload={onUpload} />);
    const file = new File([new Uint8Array([1, 2, 3])], "paper.pdf", {
      type: "application/pdf",
    });
    fireEvent.change(screen.getByLabelText("PDF file"), {
      target: { files: [file] },
    });
    fireEvent.change(screen.getByPlaceholderText("title (optional)"), {
      target: { value: "Parakeet" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Upload" }));
    await waitFor(() => expect(onUpload).toHaveBeenCalled());
    const [bytes, meta] = onUpload.mock.calls[0];
    expect(new Uint8Array(bytes)).toEqual(new Uint8Array([1, 2, 3]));
    expect(meta).toEqual({ title: "Parakeet", arxiv: "" });
  });

  it("the upload button waits for a file", () => {
    render(<CorpusPanel docs={[]} {...noop} />);
    expect(screen.getByRole("button", { name: "Upload" })).toBeDisabled();
  });

  it("a failed upload keeps the pick and shows why (§8)", async () => {
    const onUpload = vi.fn().mockRejectedValue(new Error("413: too large"));
    render(<CorpusPanel docs={[]} {...noop} onUpload={onUpload} />);
    const file = new File([new Uint8Array([1])], "big.pdf", {
      type: "application/pdf",
    });
    const input = screen.getByLabelText("PDF file");
    fireEvent.change(input, { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "Upload" }));
    expect(await screen.findByText("413: too large")).toBeInTheDocument();
    // The selection survives: the same input still holds the file, so
    // the retry is one click, not a re-pick.
    expect((input as HTMLInputElement).files?.[0]).toBe(file);
    expect(screen.getByRole("button", { name: "Upload" })).toBeEnabled();
  });
});
