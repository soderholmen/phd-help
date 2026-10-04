import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ReadView } from "./ReadView";
import type { DocSection, FilePatch } from "../types";

// Spans are the patch door's seam: start/end into the file, base the
// region's hash. `editable` is the kernel's pure-prose verdict.
const span = (start: number, editable: boolean) => ({
  start,
  end: start + 10,
  base: `hash-${start}`,
  editable,
});

const sections: DocSection[] = [
  {
    path: "main.tex",
    title: null,
    blocks: [
      { kind: "heading", level: 0, text: "The Paper", ...span(6, false) },
    ],
  },
  {
    path: "sections/a.tex",
    title: "A",
    blocks: [
      { kind: "heading", level: 1, text: "Section A", ...span(8, true) },
      {
        kind: "paragraph",
        text: "Prose with [Vaswani, 2017].",
        ...span(18, false),
      },
      { kind: "paragraph", text: "Plain prose.", ...span(47, true) },
      {
        kind: "list",
        ordered: false,
        items: ["one", "two"],
        ...span(60, false),
      },
      { kind: "list", ordered: true, items: ["first"], ...span(73, false) },
      { kind: "math", text: "\\[E=mc^2\\]", ...span(85, false) },
      { kind: "caption", text: "The curve.", ...span(96, true) },
      {
        kind: "raw",
        text: "\\begin{myenv} x \\end{myenv}",
        ...span(107, false),
      },
    ],
  },
];

const noop = () => undefined;
const noPatch = async () => {};

describe("ReadView", () => {
  it("renders every block kind as prose", () => {
    render(
      <ReadView sections={sections} onEditSource={noop} onPatch={noPatch} />,
    );
    expect(
      screen.getByRole("heading", { level: 1, name: /^The Paper/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 1, name: "Section A" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Prose with [Vaswani, 2017].").tagName).toBe(
      "P",
    );
    expect(screen.getAllByRole("listitem")).toHaveLength(3);
    expect(screen.getByText("\\[E=mc^2\\]")).toBeInTheDocument();
    expect(screen.getByText("The curve.")).toBeInTheDocument();
    expect(
      screen.getByText("\\begin{myenv} x \\end{myenv}"),
    ).toBeInTheDocument();
  });

  it("never drops: math and raw blocks show their source", () => {
    render(
      <ReadView sections={sections} onEditSource={noop} onPatch={noPatch} />,
    );
    expect(screen.getByText("\\[E=mc^2\\]").tagName).toBe("PRE");
    expect(screen.getByText("\\begin{myenv} x \\end{myenv}").tagName).toBe(
      "PRE",
    );
  });

  it("block math renders a KaTeX view above the raw source", () => {
    const { container } = render(
      <ReadView sections={sections} onEditSource={noop} onPatch={noPatch} />,
    );
    const view = container.querySelector(".math-view")!;
    // Delimiters stripped for displayMode: the annotation carries the
    // TeX KaTeX actually parsed.
    expect(view.querySelector("annotation")?.textContent).toBe("E=mc^2");
    // The raw pre stays in the DOM (never-drop): CSS hides it only
    // when a view precedes it, and jsdom proves the chain survives.
    expect(view.nextElementSibling?.textContent).toBe("\\[E=mc^2\\]");
  });

  it("unparseable math falls back to the visible raw pre", () => {
    const broken: DocSection[] = [
      {
        path: "sections/b.tex",
        title: "B",
        blocks: [
          { kind: "math", text: "\\[\\unknownmacro{x}\\]", ...span(5, false) },
        ],
      },
    ];
    const { container } = render(
      <ReadView sections={broken} onEditSource={noop} onPatch={noPatch} />,
    );
    expect(container.querySelector(".math-view")).toBeNull();
    expect(screen.getByText("\\[\\unknownmacro{x}\\]")).toBeInTheDocument();
  });

  it("inline math in a paragraph renders KaTeX spans", () => {
    const inline: DocSection[] = [
      {
        path: "sections/c.tex",
        title: "C",
        blocks: [
          {
            kind: "paragraph",
            text: "Cost $E=mc^2$ here, not \\$5.",
            ...span(7, false),
          },
        ],
      },
    ];
    const { container } = render(
      <ReadView sections={inline} onEditSource={noop} onPatch={noPatch} />,
    );
    const p = container.querySelector("p")!;
    expect(p.querySelector(".katex")).not.toBeNull();
    // The prose around the math survives, and the literal dollar
    // unescapes — it is money, not a delimiter.
    expect(p.textContent).toContain("Cost ");
    expect(p.textContent).toContain(" here, not $5.");
  });

  it("a literal dollar renders as a dollar wherever math is not split", () => {
    // The kernel keeps \$ literal for every inline text, not just
    // paragraphs — so heading/list/caption, which do no math splitting,
    // must still unescape it (a backslash there is the regression).
    const dollars: DocSection[] = [
      {
        path: "sections/d.tex",
        title: "D",
        blocks: [
          { kind: "heading", level: 1, text: "Cost \\$5 today", ...span(4, true) },
          { kind: "list", ordered: false, items: ["about \\$5"], ...span(12, false) },
          { kind: "caption", text: "Roughly \\$5.", ...span(20, true) },
        ],
      },
    ];
    render(
      <ReadView sections={dollars} onEditSource={noop} onPatch={noPatch} />,
    );
    expect(
      screen.getByRole("heading", { name: "Cost $5 today" }),
    ).toBeInTheDocument();
    expect(screen.getByText("about $5")).toBeInTheDocument();
    expect(screen.getByText("Roughly $5.")).toBeInTheDocument();
  });

  it("an empty document says so", () => {
    render(
      <ReadView sections={[]} onEditSource={noop} onPatch={noPatch} />,
    );
    expect(screen.getByText(/Nothing to read/)).toBeInTheDocument();
  });

  it("clicking an editable paragraph drafts a patch of exactly its span", async () => {
    const onPatch = vi.fn().mockResolvedValue(undefined);
    render(
      <ReadView sections={sections} onEditSource={noop} onPatch={onPatch} />,
    );
    fireEvent.click(screen.getByText("Plain prose."));
    const box = screen.getByLabelText("Edit paragraph");
    fireEvent.change(box, { target: { value: "Edited by hand." } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(onPatch).toHaveBeenCalledWith({
        path: "sections/a.tex",
        start: 47,
        end: 57,
        base: "hash-47",
        text: "Edited by hand.",
      } as FilePatch),
    );
    // A landed save closes the draft.
    await waitFor(() =>
      expect(
        screen.queryByLabelText("Edit paragraph"),
      ).not.toBeInTheDocument(),
    );
  });

  it("a bounced save keeps the draft open", async () => {
    const onPatch = vi.fn().mockRejectedValue(new Error("409"));
    render(
      <ReadView sections={sections} onEditSource={noop} onPatch={onPatch} />,
    );
    fireEvent.click(screen.getByText("Plain prose."));
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(onPatch).toHaveBeenCalled());
    expect(screen.getByLabelText("Edit paragraph")).toBeInTheDocument();
  });

  it("Discard closes the draft without a patch", () => {
    const onPatch = vi.fn();
    render(
      <ReadView sections={sections} onEditSource={noop} onPatch={onPatch} />,
    );
    fireEvent.click(screen.getByText("Plain prose."));
    fireEvent.click(screen.getByRole("button", { name: "Discard" }));
    expect(
      screen.queryByLabelText("Edit paragraph"),
    ).not.toBeInTheDocument();
    expect(onPatch).not.toHaveBeenCalled();
  });

  it("a shifted block list cannot pair the draft with another span", async () => {
    // The 3 s tick's refetch while a draft is open: an agent write
    // earlier in the file shifted every later block. The draft must
    // still post the seam it opened with — never the block now
    // rendered at its old index.
    const onPatch = vi.fn().mockResolvedValue(undefined);
    const { rerender } = render(
      <ReadView sections={sections} onEditSource={noop} onPatch={onPatch} />,
    );
    fireEvent.click(screen.getByText("Plain prose."));
    fireEvent.change(screen.getByLabelText("Edit paragraph"), {
      target: { value: "Edited by hand." },
    });
    const shifted: DocSection[] = [
      sections[0],
      {
        path: "sections/a.tex",
        title: "A",
        blocks: [
          {
            kind: "paragraph",
            text: "An agent paragraph.",
            ...span(0, true),
          },
          { kind: "heading", level: 1, text: "Section A", ...span(20, true) },
        ],
      },
    ];
    rerender(
      <ReadView sections={shifted} onEditSource={noop} onPatch={onPatch} />,
    );
    // The draft survives the shift (orphaned at the section's end).
    expect(screen.getByLabelText("Edit paragraph")).toHaveValue(
      "Edited by hand.",
    );
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(onPatch).toHaveBeenCalledWith({
        path: "sections/a.tex",
        start: 47,
        end: 57,
        base: "hash-47",
        text: "Edited by hand.",
      }),
    );
  });

  it("a citation block offers the source, not a click-edit", () => {
    const onEditSource = vi.fn();
    render(
      <ReadView
        sections={sections}
        onEditSource={onEditSource}
        onPatch={noPatch}
      />,
    );
    const para = screen.getByText("Prose with [Vaswani, 2017].");
    fireEvent.click(para); // must not open a draft
    expect(
      screen.queryByLabelText("Edit paragraph"),
    ).not.toBeInTheDocument();
    fireEvent.click(within(para).getByText("edit source"));
    expect(onEditSource).toHaveBeenCalledWith("sections/a.tex");
  });

  it("lists, math and raw offer the source too", () => {
    const onEditSource = vi.fn();
    render(
      <ReadView
        sections={sections}
        onEditSource={onEditSource}
        onPatch={noPatch}
      />,
    );
    const math = screen.getByText("\\[E=mc^2\\]");
    fireEvent.click(math.nextElementSibling!); // its edit-source link
    expect(onEditSource).toHaveBeenCalledWith("sections/a.tex");
  });

  it("Enter on a focused editable paragraph opens the draft", () => {
    // keyboard access without a role override: the heading stays a
    // heading, the paragraph stays a paragraph — tabIndex + keys.
    render(
      <ReadView sections={sections} onEditSource={noop} onPatch={noPatch} />,
    );
    fireEvent.keyDown(screen.getByText("Plain prose."), { key: "Enter" });
    expect(screen.getByLabelText("Edit paragraph")).toBeInTheDocument();
  });

  it("the draft takes focus the moment it opens", () => {
    render(
      <ReadView sections={sections} onEditSource={noop} onPatch={noPatch} />,
    );
    fireEvent.click(screen.getByText("Plain prose."));
    expect(document.activeElement).toBe(screen.getByLabelText("Edit paragraph"));
  });

  it("Escape closes an untouched draft; a typed one survives it", () => {
    render(
      <ReadView sections={sections} onEditSource={noop} onPatch={noPatch} />,
    );
    fireEvent.click(screen.getByText("Plain prose."));
    fireEvent.keyDown(screen.getByLabelText("Edit paragraph"), {
      key: "Escape",
    });
    expect(
      screen.queryByLabelText("Edit paragraph"),
    ).not.toBeInTheDocument();
    // dirty: Escape must not silently eat the user's typing — only
    // Discard closes a draft that differs from the block.
    fireEvent.click(screen.getByText("Plain prose."));
    fireEvent.change(screen.getByLabelText("Edit paragraph"), {
      target: { value: "half-typed thought" },
    });
    fireEvent.keyDown(screen.getByLabelText("Edit paragraph"), {
      key: "Escape",
    });
    expect(screen.getByLabelText("Edit paragraph")).toHaveValue(
      "half-typed thought",
    );
  });

  it("every section carries an Edit source door", () => {
    const onEditSource = vi.fn();
    render(
      <ReadView
        sections={sections}
        onEditSource={onEditSource}
        onPatch={noPatch}
      />,
    );
    const sec = screen
      .getByText("Prose with [Vaswani, 2017].")
      .closest("section")!;
    fireEvent.click(within(sec).getByRole("button", { name: "Edit source" }));
    expect(onEditSource).toHaveBeenCalledWith("sections/a.tex");
  });
});
