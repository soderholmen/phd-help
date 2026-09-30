import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { Transcript } from "./Transcript";
import type { Message } from "../protocol/types";

const messages: Message[] = [
  { id: 1, role: "user", text: "tighten it", section: "sections/intro.tex" },
  { id: 2, role: "assistant", text: "done" },
  { id: 3, role: "error", text: "llm: vLLM down" },
  { id: 4, role: "notice", text: "Applied to sections/intro.tex" },
];

it("renders every role with its class and the anchor chip", () => {
  const { container } = render(<Transcript messages={messages} />);
  expect(container.querySelector("li.msg.user")).toHaveTextContent("tighten it");
  expect(container.querySelector("li.msg.assistant")).toHaveTextContent("done");
  expect(container.querySelector("li.msg.error")).toHaveTextContent("llm: vLLM down");
  expect(container.querySelector("li.msg.notice")).toHaveTextContent("Applied");
  expect(screen.getByText("sections/intro.tex")).toBeInTheDocument();
});

it("renders nothing without messages", () => {
  const { container } = render(<Transcript messages={[]} />);
  expect(container.querySelectorAll("li")).toHaveLength(0);
});
