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

  it("Send waits for something to send", () => {
    render(<Composer onSend={() => {}} />);
    const send = screen.getByRole("button", { name: "Send" });
    expect(send).toBeDisabled();
    fireEvent.change(screen.getByLabelText("message"), {
      target: { value: "tighten it" },
    });
    expect(send).toBeEnabled();
  });

  it("a live partial shows as ghost text over the input", () => {
    const { container } = render(<Composer onSend={() => {}} partial="tighten the in" />);
    const ghost = container.querySelector(".ghost");
    expect(ghost).not.toBeNull();
    expect(ghost).toHaveTextContent("tighten the in");
    // aria-hidden: the final transcript is the accessible record —
    // the ghost is a liveness cue for the holder's eyes only.
    expect(ghost).toHaveAttribute("aria-hidden", "true");
    // The ghost is an overlay, not the placeholder: the resting hint
    // stays put.
    expect(screen.getByLabelText("message")).toHaveAttribute(
      "placeholder",
      "type, or arm voice and talk…",
    );
  });

  it("no partial, no ghost", () => {
    const { container } = render(<Composer onSend={() => {}} />);
    expect(container.querySelector(".ghost")).toBeNull();
  });

  it("the ghost yields to typed text", () => {
    const { container } = render(<Composer onSend={() => {}} partial="tighten" />);
    fireEvent.change(screen.getByLabelText("message"), { target: { value: "typed" } });
    expect(container.querySelector(".ghost")).toBeNull();
  });
});
