import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RelatedPanel } from "./RelatedPanel";
import type { RelatedEntry } from "../types";

const entry = (over: Partial<RelatedEntry>): RelatedEntry => ({
  title: "Mesh Anything",
  authors: ["Shazeer, Noam"],
  year: "2024",
  arxiv: "2401.00002",
  doi: "",
  venue: "arXiv",
  abstract: "A generative model for 2D and 3D.",
  why: "",
  found_by: "search",
  cited: false,
  in_corpus: null,
  ...over,
});

const noop = {
  onPin: () => {},
  onUnpin: () => {},
  onCite: () => {},
};

describe("RelatedPanel", () => {
  it("a card shows title, authors, the why and the abstract preview", () => {
    render(
      <RelatedPanel
        entries={[entry({ why: "the segmentation baseline" })]}
        {...noop}
      />,
    );
    expect(screen.getByText(/Mesh Anything/)).toBeInTheDocument();
    expect(screen.getByText("Shazeer, Noam")).toBeInTheDocument();
    expect(screen.getByText("the segmentation baseline")).toBeInTheDocument();
    expect(
      screen.getByText("A generative model for 2D and 3D."),
    ).toBeInTheDocument();
  });

  it("Pin rides a corpus doc only — and follows pinned_here", () => {
    const onPin = vi.fn();
    const onUnpin = vi.fn();
    const { rerender } = render(
      <RelatedPanel
        entries={[entry({ in_corpus: { doc_id: "d1", status: "indexed", pinned_here: false } })]}
        {...noop}
        onPin={onPin}
        onUnpin={onUnpin}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Pin" }));
    expect(onPin).toHaveBeenCalledWith("d1");
    rerender(
      <RelatedPanel
        entries={[entry({ in_corpus: { doc_id: "d1", status: "indexed", pinned_here: true } })]}
        {...noop}
        onPin={onPin}
        onUnpin={onUnpin}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Unpin" }));
    expect(onUnpin).toHaveBeenCalledWith("d1");
    // no doc, no Pin: the affordance would address a doc that isn't there
    rerender(<RelatedPanel entries={[entry({})]} {...noop} />);
    expect(
      screen.queryByRole("button", { name: "Pin" }),
    ).not.toBeInTheDocument();
  });

  it("Cite is hidden once the paper is cited", () => {
    const { rerender } = render(<RelatedPanel entries={[entry({})]} {...noop} />);
    expect(screen.getByRole("button", { name: "Cite" })).toBeInTheDocument();
    rerender(<RelatedPanel entries={[entry({ cited: true })]} {...noop} />);
    expect(screen.queryByRole("button", { name: "Cite" })).not.toBeInTheDocument();
    expect(screen.getByText("cited")).toBeInTheDocument();
  });

  it("Cite hands the whole entry to the caller (the typed-turn door)", () => {
    const onCite = vi.fn();
    const e = entry({});
    render(<RelatedPanel entries={[e]} {...noop} onCite={onCite} />);
    fireEvent.click(screen.getByRole("button", { name: "Cite" }));
    expect(onCite).toHaveBeenCalledWith(e);
  });

  it("the empty state points at Find when the pass is offered", () => {
    render(<RelatedPanel entries={[]} {...noop} onFind={() => {}} />);
    // the empty state itself names the pass (the button is always there
    // when offered — this asserts the guidance, not the affordance)
    expect(screen.getByText(/no papers yet/)).toHaveTextContent("Find papers");
  });

  it("without the pass, the empty state points at the agent instead", () => {
    render(<RelatedPanel entries={[]} {...noop} />);
    expect(screen.getByText(/agent/)).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Find papers" }),
    ).not.toBeInTheDocument();
  });

  it("Find starts the pass and waits while searching", () => {
    const onFind = vi.fn();
    const { rerender } = render(
      <RelatedPanel entries={[]} {...noop} onFind={onFind} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Find papers" }));
    expect(onFind).toHaveBeenCalled();
    rerender(
      <RelatedPanel entries={[]} searching {...noop} onFind={onFind} />,
    );
    expect(screen.getByRole("button", { name: /Searching/ })).toBeDisabled();
  });
});
