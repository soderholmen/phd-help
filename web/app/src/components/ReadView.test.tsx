import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ReadView } from "./ReadView";
import type { DocSection } from "../types";

const sections: DocSection[] = [
  {
    path: "main.tex",
    title: null,
    blocks: [{ kind: "heading", level: 0, text: "The Paper" }],
  },
  {
    path: "sections/a.tex",
    title: "A",
    blocks: [
      { kind: "heading", level: 1, text: "Section A" },
      { kind: "paragraph", text: "Prose with [Vaswani, 2017]." },
      { kind: "list", ordered: false, items: ["one", "two"] },
      { kind: "list", ordered: true, items: ["first"] },
      { kind: "math", text: "\\[E=mc^2\\]" },
      { kind: "caption", text: "The curve." },
      { kind: "raw", text: "\\begin{myenv} x \\end{myenv}" },
    ],
  },
];

describe("ReadView", () => {
  it("renders every block kind as prose", () => {
    render(<ReadView sections={sections} />);
    expect(
      screen.getByRole("heading", { level: 1, name: "The Paper" }),
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
    render(<ReadView sections={sections} />);
    expect(screen.getByText("\\[E=mc^2\\]").tagName).toBe("PRE");
    expect(screen.getByText("\\begin{myenv} x \\end{myenv}").tagName).toBe(
      "PRE",
    );
  });

  it("an empty document says so", () => {
    render(<ReadView sections={[]} />);
    expect(screen.getByText(/Nothing to read/)).toBeInTheDocument();
  });
});
