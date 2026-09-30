// Frame builders must match voice.py's CONTROL_TYPES contract exactly.
import { describe, expect, it } from "vitest";
import * as f from "./frames";

describe("control frames", () => {
  it("carry the shapes dispatch() reads", () => {
    expect(f.typed("hi")).toEqual({ type: "typed", text: "hi" });
    expect(f.selectSection("sections/intro.tex")).toEqual({
      type: "select_section",
      section: "sections/intro.tex",
    });
    expect(f.approve("sections/intro.tex", "d1")).toEqual({
      type: "approve",
      section: "sections/intro.tex",
      diff_id: "d1",
    });
    expect(f.reject("sections/intro.tex", "d1")).toEqual({
      type: "reject",
      section: "sections/intro.tex",
      diff_id: "d1",
    });
    expect(f.heartbeat(0.2)).toEqual({ type: "heartbeat", rms: 0.2 });
    expect(f.arm()).toEqual({ type: "arm" });
    expect(f.disarm()).toEqual({ type: "disarm" });
    expect(f.bargeIn()).toEqual({ type: "barge_in" });
  });
});
