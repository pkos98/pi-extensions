import type { Theme } from "@earendil-works/pi-coding-agent";
import { type Component, truncateToWidth, wrapTextWithAnsi } from "@earendil-works/pi-tui";
import type { SideAgentActivity } from "./side-thread.js";
import { sanitizeSingleLine } from "./text.js";

const MAX_NARRATION_LINES = 4;
const MAX_LIVE_OUTPUT_LINES = 3;
const MAX_ERROR_OUTPUT_LINES = 2;
const INDENT = " ";

/**
 * Renders side-agent tool calls and interim narration.
 * `live` shows a short output tail for the newest tool call while the turn is running.
 */
export function createAgentActivityComponent(
  getActivity: () => readonly SideAgentActivity[],
  theme: Theme,
  live: boolean,
): Component {
  return {
    render: (width) => renderAgentActivity(getActivity(), theme, width, live),
    invalidate() {},
  };
}

export function renderAgentActivity(
  activity: readonly SideAgentActivity[],
  theme: Theme,
  width: number,
  live: boolean,
): string[] {
  if (activity.length === 0 || width <= 0) return [];
  const lines: string[] = [""];
  const contentWidth = Math.max(1, width - INDENT.length * 2);
  const lastToolIndex = findLastToolIndex(activity);

  activity.forEach((entry, index) => {
    if (entry.kind === "text") {
      const wrapped = wrapTextWithAnsi(sanitizeMultiline(entry.text), contentWidth).slice(0, MAX_NARRATION_LINES);
      for (const line of wrapped) lines.push(`${INDENT}${theme.fg("muted", theme.italic(line))}`);
      return;
    }

    const { call } = entry;
    const icon =
      call.status === "running"
        ? theme.fg("warning", "…")
        : call.status === "error"
          ? theme.fg("error", "✗")
          : theme.fg("success", "✓");
    const name = theme.fg("accent", theme.bold(sanitizeSingleLine(call.name) || "tool"));
    const args = sanitizeSingleLine(call.args);
    lines.push(truncateToWidth(`${INDENT}${icon} ${name}${args ? ` ${theme.fg("muted", args)}` : ""}`, width));

    const outputLimit =
      call.status === "error"
        ? MAX_ERROR_OUTPUT_LINES
        : live && index === lastToolIndex && call.status === "running"
          ? MAX_LIVE_OUTPUT_LINES
          : 0;
    if (outputLimit === 0 || !call.output) return;
    const outputLines = call.output
      .split("\n")
      .map((line) => sanitizeSingleLine(line))
      .filter(Boolean);
    const shown = call.status === "error" ? outputLines.slice(0, outputLimit) : outputLines.slice(-outputLimit);
    const color = call.status === "error" ? "error" : "dim";
    for (const line of shown) {
      lines.push(truncateToWidth(`${INDENT}${INDENT}  ${theme.fg(color, line)}`, width));
    }
  });

  lines.push("");
  return lines;
}

function findLastToolIndex(activity: readonly SideAgentActivity[]): number {
  for (let index = activity.length - 1; index >= 0; index -= 1) {
    if (activity[index]?.kind === "tool") return index;
  }
  return -1;
}

function sanitizeMultiline(text: string): string {
  return text
    .split("\n")
    .map((line) => sanitizeSingleLine(line))
    .filter(Boolean)
    .join("\n");
}
