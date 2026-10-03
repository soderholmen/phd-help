import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Header } from "./Header";

const health = {
  vllm: "ok",
  stt: "ok",
  tts: "ok",
  corpus: "ok",
  endpoint_holder: null,
};

const noop = {
  onToggleVoice: () => {},
  onActivate: () => {},
  onNew: () => {},
  onImport: () => {},
  onDownload: () => {},
};

const base = {
  connected: true,
  armed: false,
  rms: 0,
  health,
  projects: ["my-paper", "next-paper"],
  active: "my-paper",
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Header", () => {
  it("the voice toggle still works", () => {
    const onToggleVoice = vi.fn();
    render(<Header {...base} {...noop} onToggleVoice={onToggleVoice} />);
    fireEvent.click(screen.getByRole("button", { name: /Voice: OFF/ }));
    expect(onToggleVoice).toHaveBeenCalled();
  });

  it("the project select lists every project with the active one chosen", () => {
    render(<Header {...base} {...noop} />);
    const select = screen.getByLabelText("Project") as HTMLSelectElement;
    expect(select.value).toBe("my-paper");
    expect(screen.getByRole("option", { name: "next-paper" })).toBeInTheDocument();
  });

  it("choosing a project activates it", () => {
    const onActivate = vi.fn();
    render(<Header {...base} {...noop} onActivate={onActivate} />);
    fireEvent.change(screen.getByLabelText("Project"), {
      target: { value: "next-paper" },
    });
    expect(onActivate).toHaveBeenCalledWith("next-paper");
  });

  it("New asks for a name and passes it on", () => {
    const onNew = vi.fn();
    vi.spyOn(window, "prompt").mockReturnValue(" third-paper ");
    render(<Header {...base} {...noop} onNew={onNew} />);
    fireEvent.click(screen.getByRole("button", { name: "New" }));
    expect(onNew).toHaveBeenCalledWith("third-paper");
  });

  it("New cancelled asks for nothing", () => {
    const onNew = vi.fn();
    vi.spyOn(window, "prompt").mockReturnValue(null);
    render(<Header {...base} {...noop} onNew={onNew} />);
    fireEvent.click(screen.getByRole("button", { name: "New" }));
    expect(onNew).not.toHaveBeenCalled();
  });

  it("Import takes the zip's bytes under the prompted name", async () => {
    const onImport = vi.fn();
    vi.spyOn(window, "prompt").mockReturnValue("imported");
    render(<Header {...base} {...noop} onImport={onImport} />);
    const zip = new File([new Uint8Array([7, 8])], "paper.zip");
    fireEvent.change(screen.getByLabelText("Project zip"), {
      target: { files: [zip] },
    });
    await waitFor(() => expect(onImport).toHaveBeenCalled());
    const [name, bytes] = onImport.mock.calls[0];
    expect(name).toBe("imported");
    expect(new Uint8Array(bytes)).toEqual(new Uint8Array([7, 8]));
  });

  it("Download lists every project and hands the chosen one over", () => {
    const onDownload = vi.fn();
    render(<Header {...base} {...noop} onDownload={onDownload} />);
    fireEvent.click(screen.getByRole("button", { name: "next-paper" }));
    expect(onDownload).toHaveBeenCalledWith("next-paper");
  });

  it("Import cancelled at the name prompt sends nothing", async () => {
    const onImport = vi.fn();
    vi.spyOn(window, "prompt").mockReturnValue(null);
    render(<Header {...base} {...noop} onImport={onImport} />);
    fireEvent.change(screen.getByLabelText("Project zip"), {
      target: { files: [new File([new Uint8Array([1])], "paper.zip")] },
    });
    await new Promise((r) => setTimeout(r, 0));
    expect(onImport).not.toHaveBeenCalled();
  });
});
