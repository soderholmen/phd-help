// Control frame builders — the only place frame shapes are spelled.
import type { ControlFrame } from "../types";

export const typed = (text: string): ControlFrame => ({ type: "typed", text });
export const selectSection = (section: string): ControlFrame => ({
  type: "select_section",
  section,
});
export const approve = (section: string, diffId: string): ControlFrame => ({
  type: "approve",
  section,
  diff_id: diffId,
});
export const reject = (section: string, diffId: string): ControlFrame => ({
  type: "reject",
  section,
  diff_id: diffId,
});
export const heartbeat = (rms: number): ControlFrame => ({
  type: "heartbeat",
  rms,
});
export const arm = (): ControlFrame => ({ type: "arm" });
export const disarm = (): ControlFrame => ({ type: "disarm" });
export const bargeIn = (): ControlFrame => ({ type: "barge_in" });
