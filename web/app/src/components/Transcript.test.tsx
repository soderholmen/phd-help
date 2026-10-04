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

it("the empty transcript still says how to start", () => {
  const { container } = render(<Transcript messages={[]} />);
  // an invitation, not a message: a <p>, never an <li>
  expect(screen.getByText(/nothing said yet/)).toBeInTheDocument();
  expect(container.querySelectorAll("li")).toHaveLength(0);
});

it("renders a fenced draft as a pre block with the fences gone", () => {
  const { container } = render(
    <Transcript messages={[{ id: 1, role: "assistant",
      text: "Here you go:\n\n```latex\nWe propose a hook.\n```\nSay add." }]} />);
  const pre = container.querySelector("li.msg.assistant pre");
  expect(pre).toHaveTextContent("We propose a hook.");
  expect(pre?.textContent).not.toContain("```");
  expect(container.querySelector("li.msg.assistant"))
    .toHaveTextContent("Here you go");
  expect(container.querySelector("li.msg.assistant"))
    .toHaveTextContent("Say add");
});

it("streams an unclosed draft into the pre, fence stripped", () => {
  // the draft rides the token stream: the closing fence arrives last
  const { container } = render(
    <Transcript messages={[{ id: 1, role: "assistant",
      text: "```latex\nWe propose a hook." }]} />);
  const pre = container.querySelector("li.msg.assistant pre");
  expect(pre).toHaveTextContent("We propose a hook.");
  expect(pre?.textContent).not.toContain("```");
});
