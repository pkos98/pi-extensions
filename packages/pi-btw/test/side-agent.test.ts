import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import {
  type Api,
  type AssistantMessage,
  createAssistantMessageEventStream,
  fauxAssistantMessage,
  fauxText,
  fauxToolCall,
  type Message,
  type Model,
} from "@earendil-works/pi-ai";
import { test } from "vitest";
import { createBtwTools } from "../src/btw.js";
import { normalizeBtwSettings } from "../src/settings.js";
import { completeSideAgentTurn, formatToolArgs } from "../src/side-agent.js";
import { createSideThread, type SideAgentActivity } from "../src/side-thread.js";

const model = {
  id: "side-model",
  name: "Side model",
  api: "openai-responses",
  provider: "test",
  baseUrl: "https://example.test",
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 100_000,
  maxTokens: 4096,
} as Model<Api>;

function scriptedStreamFn(steps: AssistantMessage[], requests: Message[][]): StreamFn {
  return (_model, context) => {
    requests.push(context.messages.slice());
    const message = steps.shift() ?? fauxAssistantMessage("unexpected extra request");
    const stream = createAssistantMessageEventStream();
    queueMicrotask(() => {
      stream.push({ type: "start", partial: message });
      if (message.stopReason === "error" || message.stopReason === "aborted") {
        stream.push({ type: "error", reason: message.stopReason, error: message });
      } else {
        stream.push({ type: "done", reason: message.stopReason as "stop" | "toolUse", message });
      }
    });
    return stream;
  };
}

test("tool-enabled side turn runs a tool, records activity, and keeps the agent transcript", async () => {
  const directory = await mkdtemp(join(tmpdir(), "pi-btw-agent-"));
  try {
    await writeFile(join(directory, "notes.txt"), "the answer is 42\n");
    const requests: Message[][] = [];
    const streamFn = scriptedStreamFn(
      [
        fauxAssistantMessage([fauxText("Checking the notes."), fauxToolCall("read", { path: "notes.txt" })], {
          stopReason: "toolUse",
        }),
        fauxAssistantMessage("The notes say 42."),
        fauxAssistantMessage("Still 42."),
      ],
      requests,
    );
    const thread = createSideThread("main context");
    const updates: (readonly SideAgentActivity[])[] = [];

    const first = await completeSideAgentTurn({
      thread,
      model,
      question: "What do the notes say?",
      thinkingLevel: "off",
      tools: createBtwTools(["read"], directory),
      systemPrompt: "system",
      streamFn,
      onActivity: (activity) => updates.push(activity),
    });

    assert.equal(first.kind, "answered");
    assert.equal(first.kind === "answered" && first.answer, "The notes say 42.");
    assert.equal(requests.length, 2);
    const toolResult = requests[1]?.find((message) => message.role === "toolResult");
    assert.ok(toolResult && JSON.stringify(toolResult.content).includes("the answer is 42"));
    assert.ok(JSON.stringify(requests[0]).includes("main context"));

    const turn = thread.turns[0];
    assert.equal(turn?.kind, "answered");
    const activity = turn?.kind === "answered" ? turn.activity : undefined;
    assert.deepEqual(
      activity?.map((entry) => (entry.kind === "text" ? entry.text : `${entry.call.name}:${entry.call.status}`)),
      ["Checking the notes.", "read:done"],
    );
    assert.ok(updates.length >= 2);

    const second = await completeSideAgentTurn({
      thread,
      model,
      question: "And again?",
      thinkingLevel: "off",
      tools: createBtwTools(["read"], directory),
      systemPrompt: "system",
      streamFn,
    });
    assert.equal(second.kind === "answered" && second.answer, "Still 42.");
    // The follow-up request replays the earlier tool call and its result.
    assert.ok(requests[2]?.some((message) => message.role === "toolResult"));
    assert.equal(thread.turns.length, 2);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("tool-enabled side turn reports provider errors without committing the transcript", async () => {
  const thread = createSideThread("");
  const result = await completeSideAgentTurn({
    thread,
    model,
    question: "Hi",
    thinkingLevel: "off",
    tools: [],
    systemPrompt: "system",
    streamFn: scriptedStreamFn([fauxAssistantMessage("", { stopReason: "error", errorMessage: "boom" })], []),
  });
  assert.deepEqual(result, { kind: "error", message: "boom" });
  assert.equal(thread.agentMessages, undefined);
  assert.equal(thread.turns.length, 0);
});

test("aborting a tool-enabled side turn returns aborted", async () => {
  const controller = new AbortController();
  controller.abort();
  const result = await completeSideAgentTurn({
    thread: createSideThread(""),
    model,
    question: "Hi",
    thinkingLevel: "off",
    tools: [],
    systemPrompt: "system",
    streamFn: scriptedStreamFn([], []),
    signal: controller.signal,
  });
  assert.deepEqual(result, { kind: "aborted" });
});

test("tools setting accepts known tool names and rejects unknown ones", () => {
  assert.deepEqual(normalizeBtwSettings({ tools: ["read", "bash", "read"] }), { tools: ["read", "bash"] });
  assert.deepEqual(normalizeBtwSettings({ tools: [] }), { tools: [] });
  assert.equal(normalizeBtwSettings({ tools: ["rm"] }), undefined);
  assert.equal(normalizeBtwSettings({ tools: "bash" }), undefined);
});

test("tool argument previews favor the meaningful field", () => {
  assert.equal(formatToolArgs("bash", { command: "ls -la" }), "ls -la");
  assert.equal(formatToolArgs("read", { path: "a.ts", offset: 10, limit: 5 }), "a.ts:10+5");
  assert.equal(formatToolArgs("grep", { pattern: "foo", path: "src" }), "foo in src");
  assert.equal(formatToolArgs("custom", { x: 1 }), '{"x":1}');
});

test("activity rendering sanitizes untrusted text and shows a live output tail", async () => {
  const { renderAgentActivity } = await import("../src/agent-activity.js");
  const theme = {
    fg: (_color: string, text: string) => text,
    bold: (text: string) => text,
    italic: (text: string) => text,
  };
  const activity: SideAgentActivity[] = [
    { kind: "text", text: "Looking\u001b[31m around" },
    {
      kind: "tool",
      call: { id: "1", name: "bash", args: "echo\u0007 hi", status: "error", output: "bad\nworse\nworst" },
    },
    { kind: "tool", call: { id: "2", name: "bash", args: "seq 5", status: "running", output: "1\n2\n3\n4\n5" } },
  ];
  const live = renderAgentActivity(activity, theme as never, 60, true).join("\n");
  assert.ok(!live.includes("\u0007") && !live.includes("\u001b"));
  assert.match(live, /✗ bash echo hi/u);
  assert.match(live, /bad\n.*worse/u);
  assert.doesNotMatch(live, /worst/u);
  assert.match(live, /… bash seq 5\n.*3\n.*4\n.*5/u);
  const history = renderAgentActivity(activity, theme as never, 60, false).join("\n");
  assert.doesNotMatch(history, /\b4\b/u);
});
