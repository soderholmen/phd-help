import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Composer } from "./Composer";

describe("Composer", () => {
  it("sends the trimmed text and clears the input", () => {
    const onSend = vi.fn();
    render(<Composer onSend={onSend} />);
    const input = screen.getByLabelText("message");
    fireEvent.change(input, { target: { value: "  tighten it  " } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(onSend).toHaveBeenCalledWith("tighten it");
    expect(input).toHaveValue("");
  });

  it("ignores blank input", () => {
    const onSend = vi.fn();
    render(<Composer onSend={onSend} />);
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(onSend).not.toHaveBeenCalled();
  });
});
