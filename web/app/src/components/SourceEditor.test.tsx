import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SourceEditor } from "./SourceEditor";

describe("SourceEditor", () => {
  it("opens on the loaded source", () => {
    render(
      <SourceEditor
        path="sections/a.tex"
        text={"\\section{A}\nBody.\n"}
        onSave={async () => {}}
        onCancel={() => {}}
      />,
    );
    expect(screen.getByLabelText("LaTeX source for sections/a.tex")).toHaveValue(
      "\\section{A}\nBody.\n",
    );
    expect(screen.getByText("sections/a.tex")).toBeInTheDocument();
  });

  it("Save hands the whole draft to the save door", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(
      <SourceEditor
        path="sections/a.tex"
        text={"Body.\n"}
        onSave={onSave}
        onCancel={() => {}}
      />,
    );
    fireEvent.change(screen.getByLabelText("LaTeX source for sections/a.tex"), {
      target: { value: "Edited.\n" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(onSave).toHaveBeenCalledWith("Edited.\n"));
  });

  it("a bounced save keeps the draft", async () => {
    const onSave = vi.fn().mockRejectedValue(new Error("409"));
    render(
      <SourceEditor
        path="sections/a.tex"
        text={"Body.\n"}
        onSave={onSave}
        onCancel={() => {}}
      />,
    );
    fireEvent.change(screen.getByLabelText("LaTeX source for sections/a.tex"), {
      target: { value: "Edited.\n" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(onSave).toHaveBeenCalled());
    expect(
      screen.getByLabelText("LaTeX source for sections/a.tex"),
    ).toHaveValue("Edited.\n");
  });

  it("Discard cancels without saving", () => {
    const onSave = vi.fn();
    const onCancel = vi.fn();
    render(
      <SourceEditor
        path="sections/a.tex"
        text={"Body.\n"}
        onSave={onSave}
        onCancel={onCancel}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Discard" }));
    expect(onCancel).toHaveBeenCalled();
    expect(onSave).not.toHaveBeenCalled();
  });
});
