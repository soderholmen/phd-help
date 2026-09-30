// The section tree door and the health chip source.
import type { Health, SectionNode } from "../types";
import { json } from "./http";

export const fetchTree = (): Promise<SectionNode[]> =>
  fetch("/sections").then(json<SectionNode[]>);

export const fetchHealth = (): Promise<Health> =>
  fetch("/health").then(json<Health>);
