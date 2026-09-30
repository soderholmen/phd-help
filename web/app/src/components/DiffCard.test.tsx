import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DiffCard } from "./DiffCard";
import type { DiffCard as DiffCardData } from "../protocol/types";

const card: DiffCardData = {
  diff_id: "d1",
  section: "sections/intro.tex",
  find: "old prose",
  replace: "new prose",
};

describe("DiffCard", () => {
  it("shows the raw find/replace and the target section", () => {
    render(<DiffCard card={card} onApprove={() => {}} onReject={() => {}} />);
    expect(screen.getByText("sections/intro.tex")).toBeInTheDocument();
    expect(screen.getByText("old prose").tagName).toBe("DEL");
    expect(screen.getByText("new prose").tagName).toBe("INS");
  });

  it("approve and reject carry the card through", () => {
    const onApprove = vi.fn();
    const onReject = vi.fn();
    render(<DiffCard card={card} onApprove={onApprove} onReject={onReject} />);
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));
    expect(onApprove).toHaveBeenCalledWith(card);
    fireEvent.click(screen.getByRole("button", { name: "Discard" }));
    expect(onReject).toHaveBeenCalledWith(card);
  });
});
