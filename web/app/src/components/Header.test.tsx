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
  onCommit: () => {},
  onPush: () => {},
};

const base = {
  connected: true,
  armed: false,
  rms: 0,
  health,
  projects: ["my-paper", "next-paper"],
  active: "my-paper",
  gitStatus: null,
};

const git = (over = {}) => ({
  initialized: true,
  dirty: false,
  has_remote: true,
  last: { sha: "abc1234", date: "2026-10-04T09:00:00+02:00", subject: "m" },
  ...over,
});

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

  it("the git chip tells the repo state", () => {
    render(<Header {...base} {...noop} gitStatus={git({ dirty: true })} />);
    expect(screen.getByText("● dirty")).toBeInTheDocument();
    render(
      <Header
        {...base}
        {...noop}
        gitStatus={git({ initialized: false, has_remote: false, last: null })}
      />,
    );
    expect(screen.getByText("no repo")).toBeInTheDocument();
  });

  it("Commit asks for a message and passes it on", () => {
    const onCommit = vi.fn();
    vi.spyOn(window, "prompt").mockReturnValue(" intro done ");
    render(<Header {...base} {...noop} onCommit={onCommit} />);
    fireEvent.click(screen.getByRole("button", { name: "Commit" }));
    expect(onCommit).toHaveBeenCalledWith("intro done");
  });

  it("Commit cancelled asks for nothing", () => {
    const onCommit = vi.fn();
    vi.spyOn(window, "prompt").mockReturnValue(null);
    render(<Header {...base} {...noop} onCommit={onCommit} />);
    fireEvent.click(screen.getByRole("button", { name: "Commit" }));
    expect(onCommit).not.toHaveBeenCalled();
  });

  it("Push without a remote asks for the URL first", () => {
    const onPush = vi.fn();
    vi.spyOn(window, "prompt").mockReturnValue("https://github.com/u/r.git");
    render(
      <Header {...base} {...noop} onPush={onPush} gitStatus={git({ has_remote: false })} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Push" }));
    expect(onPush).toHaveBeenCalledWith("https://github.com/u/r.git");
  });

  it("Push with a remote pushes directly", () => {
    const onPush = vi.fn();
    render(<Header {...base} {...noop} onPush={onPush} gitStatus={git()} />);
    fireEvent.click(screen.getByRole("button", { name: "Push" }));
    expect(onPush).toHaveBeenCalledWith();
  });

  it("Undo clicks straight through — no confirm dialog", () => {
    const onUndo = vi.fn();
    const prompt = vi.spyOn(window, "prompt");
    const { unmount } = render(<Header {...base} {...noop} />);
    expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
    unmount();
    render(<Header {...base} {...noop} onUndo={onUndo} />);
    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(onUndo).toHaveBeenCalled();
    expect(prompt).not.toHaveBeenCalled(); // direct write, low blast radius
  });

  it("the theme toggle appears only when wired, and flips on click", () => {
    const onToggleTheme = vi.fn();
    const { unmount } = render(<Header {...base} {...noop} />);
    expect(
      screen.queryByRole("button", { name: /theme/i }),
    ).toBeNull(); // unwired (tests, embeds): no dead button
    unmount();
    render(
      <Header
        {...base}
        {...noop}
        theme="dark"
        onToggleTheme={onToggleTheme}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /theme/i }));
    expect(onToggleTheme).toHaveBeenCalled();
  });

  it("Push cancelled at the URL prompt pushes nothing", () => {
    const onPush = vi.fn();
    vi.spyOn(window, "prompt").mockReturnValue(null);
    render(
      <Header {...base} {...noop} onPush={onPush} gitStatus={git({ has_remote: false })} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Push" }));
    expect(onPush).not.toHaveBeenCalled();
  });
});
