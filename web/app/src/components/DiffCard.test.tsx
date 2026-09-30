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

  it("approve carries the card through", () => {
    const onApprove = vi.fn();
    render(<DiffCard card={card} onApprove={onApprove} onReject={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));
    expect(onApprove).toHaveBeenCalledWith(card);
  });

  it("reject carries the card through", () => {
    const onReject = vi.fn();
    render(<DiffCard card={card} onApprove={() => {}} onReject={onReject} />);
    fireEvent.click(screen.getByRole("button", { name: "Discard" }));
    expect(onReject).toHaveBeenCalledWith(card);
  });

  it("one decision per card: both buttons go dead after a click", () => {
    const onApprove = vi.fn();
    const onReject = vi.fn();
    render(<DiffCard card={card} onApprove={onApprove} onReject={onReject} />);
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));
    expect(screen.getByRole("button", { name: "Apply" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Discard" })).toBeDisabled();
    // A second frame of the same click changes nothing.
    fireEvent.click(screen.getByRole("button", { name: "Discard" }));
    expect(onReject).not.toHaveBeenCalled();
    expect(onApprove).toHaveBeenCalledTimes(1);
  });
});
