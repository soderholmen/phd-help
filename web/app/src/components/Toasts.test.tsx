import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Toasts } from "./Toasts";

// The toast is the notice surface the Read view never had: a notice
// nobody can see is §8's failure mode.

describe("Toasts", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("a notice rides role=status with its text", () => {
    render(
      <Toasts
        toast={{ id: 1, kind: "notice", text: "Applied to intro" }}
        onDismiss={() => {}}
      />,
    );
    expect(screen.getByRole("status")).toHaveTextContent("Applied to intro");
  });

  it("an error rides role=alert", () => {
    render(
      <Toasts
        toast={{ id: 2, kind: "error", text: "Push failed" }}
        onDismiss={() => {}}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Push failed");
  });

  it("nothing toasts without a toast", () => {
    const { container } = render(<Toasts toast={null} onDismiss={() => {}} />);
    expect(container.firstChild).toBeNull();
  });

  it("Dismiss hands the gesture out", () => {
    const onDismiss = vi.fn();
    render(
      <Toasts toast={{ id: 1, kind: "notice", text: "x" }} onDismiss={onDismiss} />,
    );
    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(onDismiss).toHaveBeenCalled();
  });

  it("a toast dismisses itself after 6 s", () => {
    const onDismiss = vi.fn();
    render(
      <Toasts toast={{ id: 1, kind: "notice", text: "x" }} onDismiss={onDismiss} />,
    );
    act(() => vi.advanceTimersByTime(6000));
    expect(onDismiss).toHaveBeenCalled();
  });
});
