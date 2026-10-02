import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FilesPanel } from "./FilesPanel";
import type { ProjectFile } from "../api/files";

const noop = {
  onUpload: async () => {},
  onRemove: () => {},
  onRestore: () => {},
};

const file = (path: string, linked: boolean): ProjectFile => ({
  path,
  linked,
});

describe("FilesPanel", () => {
  it("lists files, marks the unlinked ones", () => {
    render(
      <FilesPanel
        files={[file("sections/intro.tex", true), file("sections/orphan.tex", false)]}
        trash={[]}
        {...noop}
      />,
    );
    expect(screen.getByText("sections/intro.tex")).toBeInTheDocument();
    expect(screen.getByText("(not in the document)")).toBeInTheDocument();
  });

  it("remove names the file it soft-deletes", () => {
    const onRemove = vi.fn();
    render(
      <FilesPanel files={[file("sections/intro.tex", true)]} trash={[]} {...noop} onRemove={onRemove} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    expect(onRemove).toHaveBeenCalledWith("sections/intro.tex");
  });

  it("trash entries get a restore that names the path", () => {
    const onRestore = vi.fn();
    render(
      <FilesPanel files={[]} trash={["sections/intro.tex"]} {...noop} onRestore={onRestore} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));
    expect(onRestore).toHaveBeenCalledWith("sections/intro.tex");
  });

  it("uploads the picked file's bytes under its own name", async () => {
    const onUpload = vi.fn();
    render(<FilesPanel files={[]} trash={[]} {...noop} onUpload={onUpload} />);
    const picked = new File([new Uint8Array([4, 5])], "method.tex", {
      type: "application/x-tex",
    });
    fireEvent.change(screen.getByLabelText("TeX file"), {
      target: { files: [picked] },
    });
    fireEvent.click(screen.getByRole("button", { name: "Upload" }));
    await waitFor(() => expect(onUpload).toHaveBeenCalled());
    const [name, bytes] = onUpload.mock.calls[0];
    expect(name).toBe("method.tex");
    expect(new Uint8Array(bytes)).toEqual(new Uint8Array([4, 5]));
  });

  it("a failed upload keeps the pick and shows why (§8)", async () => {
    const onUpload = vi.fn().mockRejectedValue(new Error("file already exists"));
    render(<FilesPanel files={[]} trash={[]} {...noop} onUpload={onUpload} />);
    const picked = new File([new Uint8Array([1])], "intro.tex");
    const input = screen.getByLabelText("TeX file");
    fireEvent.change(input, { target: { files: [picked] } });
    fireEvent.click(screen.getByRole("button", { name: "Upload" }));
    expect(await screen.findByText("file already exists")).toBeInTheDocument();
    expect((input as HTMLInputElement).files?.[0]).toBe(picked);
  });
});
