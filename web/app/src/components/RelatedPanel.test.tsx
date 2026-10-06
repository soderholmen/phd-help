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
  query: "",
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

  it("a long abstract clamps to a preview and more opens it in place", () => {
    const long = "A long abstract about mesh generation. ".repeat(12).trim();
    render(<RelatedPanel entries={[entry({ abstract: long })]} {...noop} />);
    fireEvent.click(screen.getByRole("button", { name: "more" }));
    expect(screen.getByRole("button", { name: "less" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "less" }));
    expect(screen.getByRole("button", { name: "more" })).toBeInTheDocument();
  });

  it("a short abstract needs no more button", () => {
    render(
      <RelatedPanel
        entries={[entry({ abstract: "Short and complete." })]}
        {...noop}
      />,
    );
    expect(
      screen.queryByRole("button", { name: "more" }),
    ).not.toBeInTheDocument();
  });

  it("Open links to the arXiv page, and to the DOI when there is no id", () => {
    const { rerender } = render(
      <RelatedPanel entries={[entry({})]} {...noop} />,
    );
    expect(screen.getByRole("link", { name: "Open" })).toHaveAttribute(
      "href",
      "https://arxiv.org/abs/2401.00002",
    );
    rerender(
      <RelatedPanel
        entries={[entry({ arxiv: "", doi: "10.1000/x" })]}
        {...noop}
      />,
    );
    expect(screen.getByRole("link", { name: "Open" })).toHaveAttribute(
      "href",
      "https://doi.org/10.1000/x",
    );
    // no id, nothing to open — a dead link is worse than none
    rerender(
      <RelatedPanel entries={[entry({ arxiv: "", doi: "" })]} {...noop} />,
    );
    expect(
      screen.queryByRole("link", { name: "Open" }),
    ).not.toBeInTheDocument();
  });

  it("a card names the query that found it", () => {
    render(
      <RelatedPanel entries={[entry({ query: "maritime jcf" })]} {...noop} />,
    );
    expect(screen.getByText(/maritime jcf/)).toBeInTheDocument();
  });

  it("the steer box hands keywords and focus to Find", () => {
    const onFind = vi.fn();
    render(<RelatedPanel entries={[]} {...noop} onFind={onFind} />);
    fireEvent.change(screen.getByLabelText("Keywords"), {
      target: { value: "maritime jcf, ship traffic" },
    });
    fireEvent.change(screen.getByLabelText("Where to search"), {
      target: { value: "IEEE venues" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Find papers" }));
    expect(onFind).toHaveBeenCalledWith(
      ["maritime jcf", "ship traffic"],
      "IEEE venues",
    );
  });

  it("the header says what the last pass searched for", () => {
    render(
      <RelatedPanel
        entries={[]}
        {...noop}
        lastPass={{
          keywords: ["maritime jcf"],
          focus: "IEEE venues",
          at: "2026-10-05 19:40",
        }}
      />,
    );
    expect(screen.getByText(/maritime jcf/)).toBeInTheDocument();
    expect(screen.getByText(/IEEE venues/)).toBeInTheDocument();
  });
});
