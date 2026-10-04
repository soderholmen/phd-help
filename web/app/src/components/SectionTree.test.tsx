import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SectionTree } from "./SectionTree";
import type { SectionNode } from "../types";

const tree: SectionNode[] = [
  {
    path: "sections/intro.tex",
    title: "Introduction",
    children: [
      { path: "sections/background.tex", title: "Background", children: [] },
    ],
  },
  { path: "sections/notes.tex", title: null, children: [] },
];

describe("SectionTree", () => {
  it("renders titles, falling back to the path when untitled", () => {
    render(<SectionTree tree={tree} selected={null} onSelect={() => {}} />);
    expect(screen.getByText("Introduction")).toBeInTheDocument();
    expect(screen.getByText("Background")).toBeInTheDocument();
    expect(screen.getByText("sections/notes.tex")).toBeInTheDocument();
  });

  it("clicking a node at any depth selects its path", () => {
    const onSelect = vi.fn();
    render(<SectionTree tree={tree} selected={null} onSelect={onSelect} />);
    fireEvent.click(screen.getByText("Background"));
    expect(onSelect).toHaveBeenCalledWith("sections/background.tex");
  });

  it("marks the selected node and offers the clear", () => {
    const onSelect = vi.fn();
    render(
      <SectionTree tree={tree} selected="sections/intro.tex" onSelect={onSelect} />,
    );
    expect(screen.getByText("Introduction")).toHaveAttribute("aria-current", "true");
    fireEvent.click(screen.getByRole("button", { name: /clear anchor/i }));
    expect(onSelect).toHaveBeenCalledWith("");
  });

  it("hides the clear control when nothing is anchored", () => {
    render(<SectionTree tree={tree} selected={null} onSelect={() => {}} />);
    expect(screen.queryByRole("button", { name: /clear anchor/i })).toBeNull();
  });

  it("an empty tree says so without faking a node", () => {
    const { container } = render(
      <SectionTree tree={[]} selected={null} onSelect={() => {}} />,
    );
    expect(screen.getByText(/no sections yet/)).toBeInTheDocument();
    expect(container.querySelectorAll("li")).toHaveLength(0);
  });
});
