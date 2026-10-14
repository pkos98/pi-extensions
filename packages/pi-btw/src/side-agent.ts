import {
  Agent,
  type AgentEvent,
  type AgentMessage,
  type AgentTool,
  type StreamFn,
} from "@earendil-works/pi-agent-core";
import type { Api, AssistantMessage, Model } from "@earendil-works/pi-ai";
import {
  type BtwThinkingLevel,
  buildFollowUpPrompt,
  buildSideThreadMessages,
  type CompleteSideThreadTurnResult,
  createUserMessage,
  extractAssistantText,
  type SideAgentActivity,
  type SideThread,
  type SideToolCallSummary,
} from "./side-thread.js";

const MAX_TOOL_OUTPUT_CHARS = 4000;
const MAX_TOOL_ARGS_CHARS = 400;

export const BTW_AGENT_SYSTEM_PROMPT_NOTE = `# /btw side thread

You are running in a /btw side thread next to the user's main coding-agent session. The main agent may be working at the same time.
Answer the user's side question directly and concisely. Use your tools when they help answer the question, for example to inspect files or run commands.
Only the tools declared to you in this side thread are callable; ignore any other tools mentioned above.
Prefer read-only investigation. Change files or run commands with side effects only when the user asks for it, and keep in mind that the main agent may be editing the same files.
The main conversation is provided as background in the first message; you cannot change it.`;

export interface CompleteSideAgentTurnOptions {
  thread: SideThread;
  model: Model<Api>;
  question: string;
  thinkingLevel: BtwThinkingLevel;
  tools: AgentTool[];
  systemPrompt: string;
  streamFn: StreamFn;
  signal?: AbortSignal;
  onActivity?: (activity: readonly SideAgentActivity[]) => void;
}

/** Run one side-thread turn through Pi's agent loop so the model can call tools before answering. */
export async function completeSideAgentTurn({
  thread,
  model,
  question,
  thinkingLevel,
  tools,
  systemPrompt,
  streamFn,
  signal,
  onActivity,
}: CompleteSideAgentTurnOptions): Promise<CompleteSideThreadTurnResult> {
  if (signal?.aborted) return { kind: "aborted" };

  let history: AgentMessage[];
  let prompt: AgentMessage;
  if (thread.agentMessages) {
    history = thread.agentMessages;
    prompt = createUserMessage(buildFollowUpPrompt(question));
  } else {
    // Answer-only turns (or a fresh thread) seed the transcript the same way the single-call path does.
    const messages = buildSideThreadMessages(thread, question);
    history = messages.slice(0, -1);
    prompt = messages.at(-1) ?? createUserMessage(question);
  }

  const agent = new Agent({
    initialState: { systemPrompt, model, thinkingLevel, tools, messages: history },
    streamFn,
    sessionId: thread.routingSessionId,
  });
  const activity: SideAgentActivity[] = [];
  const calls = new Map<string, SideToolCallSummary>();
  const publish = () => onActivity?.(activity.slice());
  const unsubscribe = agent.subscribe((event) => {
    if (recordActivity(event, activity, calls)) publish();
  });
  const abort = () => agent.abort();
  signal?.addEventListener("abort", abort, { once: true });

  try {
    await agent.prompt(prompt);
    await agent.waitForIdle();
  } catch (error: unknown) {
    if (signal?.aborted) return { kind: "aborted" };
    return { kind: "error", message: error instanceof Error ? error.message : String(error) };
  } finally {
    signal?.removeEventListener("abort", abort);
    unsubscribe();
  }

  if (signal?.aborted) return { kind: "aborted" };
  const response = findLastAssistantMessage(agent.state.messages);
  if (!response) return { kind: "error", message: agent.state.errorMessage ?? "The side agent returned no response." };
  if (response.stopReason === "aborted") return { kind: "aborted" };
  if (response.stopReason === "error") {
    return { kind: "error", message: response.errorMessage ?? agent.state.errorMessage ?? "The side agent failed." };
  }

  const answer = extractAssistantText(response) || "No response received.";
  thread.agentMessages = agent.state.messages.slice();
  thread.turns.push({
    kind: "answered",
    question,
    answer,
    response,
    ...(activity.length > 0 ? { activity: activity.slice() } : {}),
  });
  return { kind: "answered", response, answer };
}

function recordActivity(
  event: AgentEvent,
  activity: SideAgentActivity[],
  calls: Map<string, SideToolCallSummary>,
): boolean {
  switch (event.type) {
    case "message_end": {
      const message = event.message;
      if (message.role !== "assistant") return false;
      // Interim narration only: the final answer is rendered as the assistant reply.
      if (!message.content.some((content) => content.type === "toolCall")) return false;
      const text = extractAssistantText(message);
      if (!text) return false;
      activity.push({ kind: "text", text });
      return true;
    }
    case "tool_execution_start": {
      const call: SideToolCallSummary = {
        id: event.toolCallId,
        name: event.toolName,
        args: formatToolArgs(event.toolName, event.args),
        status: "running",
      };
      calls.set(call.id, call);
      activity.push({ kind: "tool", call });
      return true;
    }
    case "tool_execution_update": {
      const call = calls.get(event.toolCallId);
      if (!call) return false;
      call.output = formatToolOutput(event.partialResult);
      return true;
    }
    case "tool_execution_end": {
      const call = calls.get(event.toolCallId);
      if (!call) return false;
      call.status = event.isError ? "error" : "done";
      call.output = formatToolOutput(event.result);
      return true;
    }
    default:
      return false;
  }
}

function findLastAssistantMessage(messages: readonly AgentMessage[]): AssistantMessage | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.role === "assistant") return message;
  }
  return undefined;
}

export function formatToolArgs(toolName: string, args: unknown): string {
  if (args === null || typeof args !== "object") return truncate(String(args ?? ""), MAX_TOOL_ARGS_CHARS);
  const record = args as Record<string, unknown>;
  const pick = (key: string) => (typeof record[key] === "string" ? (record[key] as string) : undefined);
  let preview: string | undefined;
  if (toolName === "bash") preview = pick("command");
  else if (toolName === "read") {
    const path = pick("path");
    const offset = typeof record.offset === "number" ? `:${record.offset}` : "";
    const limit = typeof record.limit === "number" ? `+${record.limit}` : "";
    preview = path === undefined ? undefined : `${path}${offset}${limit}`;
  } else if (toolName === "edit" || toolName === "write" || toolName === "ls") preview = pick("path");
  else if (toolName === "grep" || toolName === "find") {
    const pattern = pick("pattern");
    const path = pick("path");
    preview = pattern === undefined ? undefined : path ? `${pattern} in ${path}` : pattern;
  }
  return truncate(preview ?? safeJson(args), MAX_TOOL_ARGS_CHARS);
}

export function formatToolOutput(result: unknown): string {
  if (result === null || typeof result !== "object") return truncate(String(result ?? ""), MAX_TOOL_OUTPUT_CHARS);
  const content = (result as { content?: unknown }).content;
  if (!Array.isArray(content)) return truncate(safeJson(result), MAX_TOOL_OUTPUT_CHARS);
  const text = content
    .map((block: unknown) => {
      if (block === null || typeof block !== "object") return "";
      const typed = block as { type?: unknown; text?: unknown };
      if (typed.type === "text" && typeof typed.text === "string") return typed.text;
      if (typed.type === "image") return "[image]";
      return "";
    })
    .filter(Boolean)
    .join("\n");
  return truncate(text, MAX_TOOL_OUTPUT_CHARS);
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value) ?? "";
  } catch {
    return String(value);
  }
}

function truncate(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`;
}
