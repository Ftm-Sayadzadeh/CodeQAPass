import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { loadPreferences } from "./preferences";

vi.mock("@monaco-editor/react", () => ({
  default: ({ value }: { value: string }) => <pre data-testid="monaco">{value}</pre>,
}));

const project = { id: 1, name: "MarkupSafe", created_at: "2026-01-01", updated_at: "2026-01-01", files: 1, symbols: 1, chunks: 1, vector_complete: true };
const file = { id: 4, relative_path: "src/markupsafe/__init__.py", size_bytes: 4000, sha256: "abc", status: "indexed" };
const symbol = { id: 12, file_id: 4, kind: "function", name: "escape_silent", qualified_name: "escape_silent", is_async: false, start_line: 48, end_line: 61, parameters: ["s"], returns: "Markup" };
const citation = { file_id: 4, symbol_id: 12, chunk_id: "chunk-escape", source_file: file.relative_path, symbol_name: "escape_silent", qualified_name: "escape_silent", start_line: 48, end_line: 61 };
const source = { id: 4, relative_path: file.relative_path, sha256: "abc", content: "\n".repeat(47) + "def escape_silent(s):\n    if s is None:\n        return Markup()\n" };
const evaluation = {
  scope: "benchmark_evaluation",
  not_per_answer_confidence: true,
  artifact_sha256: "baseline",
  data: {
    benchmark: { questions: 60, concepts: 30 },
    questions: [{ id: "official-en", task: "search", repository: "example/repo", language: "en", category: "function_behavior", question: "How does escaping work?" }],
    aggregates: [
      { slice: { kind: "global_micro", value: "all" }, method: "lexical", questions: 60, top_1: 0.43, top_3: 0.71, mrr_at_10: 0.58 },
      { slice: { kind: "global_micro", value: "all" }, method: "semantic", questions: 60, top_1: 0.35, top_3: 0.65, mrr_at_10: 0.5 },
      { slice: { kind: "global_micro", value: "all" }, method: "hybrid", questions: 60, top_1: 0.633, top_3: 0.783, mrr_at_10: 0.732 },
      { slice: { kind: "language", value: "fa" }, method: "lexical", questions: 30, top_1: 0.5, top_3: 0.733, mrr_at_10: 0.631 },
      { slice: { kind: "language", value: "fa" }, method: "semantic", questions: 30, top_1: 0.2, top_3: 0.533, mrr_at_10: 0.377 },
      { slice: { kind: "language", value: "fa" }, method: "hybrid", questions: 30, top_1: 0.567, top_3: 0.767, mrr_at_10: 0.678 },
      { slice: { kind: "language", value: "en" }, method: "lexical", questions: 30, top_1: 0.367, top_3: 0.7, mrr_at_10: 0.53 },
      { slice: { kind: "language", value: "en" }, method: "semantic", questions: 30, top_1: 0.5, top_3: 0.767, mrr_at_10: 0.636 },
      { slice: { kind: "language", value: "en" }, method: "hybrid", questions: 30, top_1: 0.7, top_3: 0.8, mrr_at_10: 0.787 },
    ],
  },
};
const performance = {
  scope: "benchmark_evaluation",
  not_per_answer_confidence: true,
  artifact_sha256: "performance",
  data: {
    measurement_context: "descriptive measurements from the recorded evaluation environment",
    aggregates: [
      { slice: { kind: "method", method: "lexical" }, samples: 300, latency_ms: { p95: 143.9 } },
      { slice: { kind: "method", method: "semantic" }, samples: 300, latency_ms: { p95: 174.1 } },
      { slice: { kind: "method", method: "hybrid" }, samples: 300, latency_ms: { p95: 283.1 } },
    ],
  },
};
const finalThesisEvaluation = {
  scope: "benchmark_evaluation", not_per_answer_confidence: true, artifact_sha256: "thesis",
  data: {
    evaluation_id: "final_thesis_evaluation_v1", index_complete: true,
    design: { repositories: 3, search_queries: 36, qa_cases: 12, documentation_cases: 9 },
    models: { embeddings: { nomic: "nomic", gemini_001: "gemini-embedding-001", gemini_2: "gemini-embedding-2" }, llms: { qwen: "Qwen", glm: "GLM" } },
    search: { records: 324, global: Object.fromEntries(["nomic", "gemini_001", "gemini_2"].map((name, index) => [name, {
      semantic: { cases: 36, hit_at_1: 10 + index, hit_at_3: 20 + index, hit_at_5: 25, hit_at_10: 30, mrr_at_10: .5 + index / 10 },
      hybrid: { cases: 36, hit_at_1: 12, hit_at_3: 22, hit_at_5: 26, hit_at_10: 31, mrr_at_10: .6 },
    }])) },
    qa: {
      execution: { overall: { initial_success: 51, recovered_by_retry_1: 9, recovered_by_retry_2: 11, final_failure: 1, total: 72 } },
      quality: { qa_by_llm: Object.fromEntries(["qwen", "glm"].map((name, index) => [name, { scored_records: 35 + index, correctness_0_10: { n: 35, mean: 7 + index }, groundedness_0_10: { n: 35, mean: 7.5 + index }, persian_readability_0_10: { n: 17, mean: 5 + index * 3 }, usefulness_0_10: { n: 35, mean: 7 + index } }])) },
      paired_effects: { glm_minus_qwen: { paired_cases: 35, treatment_minus_control: { correctness_0_10: { n: 35, mean: 1 }, persian_readability_0_10: { n: 17, mean: 2.8 } } } },
    },
    documentation: { execution: { expected_records: 18, actual_records: 18, unique_records: 18, glm_complete: 9, glm_failed: 0, qwen_complete: 0, qwen_failed: 9, citation_identity_mismatches: 0 }, final_status: { usable_documentation_outputs: 9, unavailable_documentation_outputs: 9, quality_interpretation: "Only available outputs are scored." }, quality: { by_llm: { glm: { scored_records: 9, correctness_0_10: { n: 9, mean: 6.8 }, groundedness_0_10: { n: 9, mean: 8.2 }, persian_readability_0_10: { n: 9, mean: 8.1 }, usefulness_0_10: { n: 9, mean: 8.1 } } } } },
    human_evaluation: { overall: { scored_records: 80, correctness_0_10: { n: 80, mean: 7.4 }, groundedness_0_10: { n: 80, mean: 7.7 }, persian_readability_0_10: { n: 44, mean: 6.9 }, usefulness_0_10: { n: 80, mean: 7.4 } }, records: 90, usable: 80, unavailable: 10, limitations: [] },
    questions: [{ id: "FTE-QA-1", task: "qa", repository: "codecompass", language: "fa", difficulty: "medium", question: "بازیابی ترکیبی چگونه کار می‌کند؟" }],
    qa_details: [{
      case_id: "FTE-QA-1", embedding_arm: "gemini_2", llm_arm: "glm", execution_provenance: "retry_2", execution_status: "complete", answer: "Grounded frozen answer.",
      citations: [{ source_file: "src/example.py", qualified_symbol: "example", start_line: 1, end_line: 3 }],
      human_scores: { correctness_0_10: 9, groundedness_0_10: 8, persian_readability_0_10: 7, usefulness_0_10: 8, hallucination: "خیر" },
    }, {
      case_id: "FTE-QA-1", embedding_arm: "nomic", llm_arm: "qwen", execution_provenance: "initial", execution_status: "failed", answer: null, citations: [],
      human_scores: { correctness_0_10: null, groundedness_0_10: null, persian_readability_0_10: null, usefulness_0_10: null, hallucination: null },
    }],
  },
};
const contextStrategyEvaluation = {
  scope: "benchmark_evaluation", not_per_answer_confidence: true, artifact_sha256: "context-strategy",
  data: {
    evaluation_id: "whole_repo_rag_human_validation_v1",
    review: { reviewer: "independent_anonymous_human", blinded_to_method_labels: true, completed_at: "2026-09-09", unique_responses: 54, pairs_per_comparison: 12, missing_ratings: 0 },
    comparisons: {
      semantic_vs_whole_repo: { pairs: 12, metrics: { quality: { semantic_mean: 8.583, whole_mean: 9.958, delta: -1.375, ci95: [-2.979, -.187], p_exact: .031 } } },
      semantic_vs_lexical: { pairs: 12, metrics: { quality: { semantic_mean: 8.771, lexical_mean: 4.521, delta: 4.25, ci95: [2.396, 6.104], p_exact: .004 } } },
      semantic_vs_git_agent: { pairs: 12, metrics: { quality: { semantic_mean: 8.771, agent_mean: 7.833, delta: .938, ci95: [-.875, 2.938], p_exact: .309 } } },
    },
    limitations: ["single reviewer"],
  },
};

function response(body: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
}

function installApi(handler?: (path: string, init?: RequestInit) => Promise<Response> | undefined) {
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input).replace("/api", "");
    const custom = handler?.(path, init);
    if (custom) return custom;
    if (path === "/projects") return response([project]);
    if (path === "/projects/1") return response(project);
    if (path === "/projects/1/files") return response([file]);
    if (path === "/projects/1/symbols") return response([symbol]);
    if (path === "/projects/1/files/4/content") return response(source);
    if (path === "/projects/index-jobs/active") return Promise.resolve(new Response(null, { status: 204 }));
    if (path === "/evaluation/summary") return response(evaluation);
    if (path === "/evaluation/performance") return response(performance);
    if (path === "/evaluation/final-thesis") return response(finalThesisEvaluation);
    if (path === "/evaluation/context-strategy") return response(contextStrategyEvaluation);
    throw new Error(`Unhandled request: ${path}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function ready() {
  expect(await screen.findByText("MarkupSafe")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Show results" }));
  expect((await screen.findAllByText("63.3%")).length).toBeGreaterThan(0);
}

describe("CodeQAPass SPA", () => {
  it("opens the styled project menu and closes it after selection", async () => {
    installApi();
    render(<App />);
    await ready();

    const question = screen.getByLabelText("Ask about this codebase");
    expect(question).toHaveAttribute("rows", "2");
    expect(question.closest(".workspace-panel")).toHaveClass("empty");
    expect(screen.getByText("Advanced").closest("details")?.parentElement).toHaveClass("ask-controls");
    expect(screen.queryByText("PY")).not.toBeInTheDocument();
    expect(screen.queryByText("Where is input validation handled?")).not.toBeInTheDocument();
    expect(screen.queryByText("منطق اصلی پروژه کجاست؟")).not.toBeInTheDocument();
    const explorer = screen.getByLabelText("Project explorer");
    expect(within(explorer).getByText("src")).toBeInTheDocument();
    expect(within(explorer).getByText("markupsafe")).toBeInTheDocument();
    expect(within(explorer).getByRole("button", { name: file.relative_path })).toHaveTextContent("__init__.py");

    fireEvent.click(screen.getByRole("tab", { name: "Search" }));
    expect(screen.getByLabelText("Search indexed code").closest(".workspace-panel")).toHaveClass("empty");
    fireEvent.click(screen.getByRole("tab", { name: "Documentation" }));
    expect(screen.getByLabelText("Function or method").closest(".workspace-panel")).toHaveClass("empty");

    fireEvent.click(screen.getByRole("button", { name: "Current project: MarkupSafe" }));
    expect(screen.getByRole("menu", { name: "Projects" })).toBeInTheDocument();
    expect(screen.getByRole("menuitemradio", { name: "MarkupSafe" })).toHaveTextContent("MarkupSafe");
    expect(screen.getByRole("menuitemradio", { name: "MarkupSafe" })).not.toHaveTextContent(/files|symbols/i);
    expect(screen.getByRole("menuitemradio", { name: /MarkupSafe/ })).toHaveAttribute("aria-checked", "true");
    fireEvent.click(screen.getByRole("menuitemradio", { name: /MarkupSafe/ }));
    expect(screen.queryByRole("menu", { name: "Projects" })).not.toBeInTheDocument();
  });

  let stored: Map<string, string>;

  beforeEach(() => {
    stored = new Map();
    Object.defineProperty(window, "localStorage", { value: {
      setItem: vi.fn((key: string, value: string) => stored.set(key, value)),
      getItem: vi.fn((key: string) => stored.get(key) ?? null),
      removeItem: vi.fn((key: string) => stored.delete(key)),
      clear: vi.fn(() => stored.clear()),
    }, configurable: true });
    Object.defineProperty(window, "sessionStorage", { value: { setItem: vi.fn(), getItem: vi.fn(), removeItem: vi.fn(), clear: vi.fn() }, configurable: true });
  });

  it("renders API-driven project/evaluation data and keeps API keys memory-only", async () => {
    installApi();
    render(<App />);
    await ready();

    expect(screen.getByText("Benchmark evaluation results — not per-answer confidence.")).toBeInTheDocument();
    expect(screen.getByText("Scientific evaluation")).toBeInTheDocument();
    expect(screen.queryByText("Frozen scientific evidence")).not.toBeInTheDocument();
    expect(screen.getByText("Best measured").closest(".method-label")).toHaveTextContent("hybrid");
    expect(screen.getByText("283.1 ms")).toBeInTheDocument();
    expect(screen.getByText(/Hybrid retrieval was strongest overall/)).toBeInTheDocument();
    fireEvent.click(screen.getByText(/View 1 benchmark questions/));
    expect(screen.getByText("How does escaping work?")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show Persian evaluation" }));
    expect(screen.getAllByText("56.7%").length).toBeGreaterThan(0);
    expect(screen.getByText("Persian Top-3").nextSibling).toHaveTextContent("76.7%");
    fireEvent.click(screen.getByRole("button", { name: "Compare Persian and English evaluation" }));
    expect(screen.getByRole("region", { name: "Persian retrieval results" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "English retrieval results" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Final thesis evaluation" }));
    expect(screen.getByText("3 repositories · 36 search queries · 90 human-review records")).toBeInTheDocument();
    expect(screen.getByText(/There was no universal model winner/)).toBeInTheDocument();
    expect(screen.getByText("71/72")).toBeInTheDocument();
    fireEvent.click(screen.getByText(/View 1 benchmark questions/));
    expect(screen.getByText("بازیابی ترکیبی چگونه کار می‌کند؟")).toBeInTheDocument();
    const qaRuns = screen.getByText("2 evaluated configurations").parentElement!;
    fireEvent.click(within(qaRuns).getByText("gemini-embedding-2"));
    expect(within(qaRuns).getByText("Grounded frozen answer.")).toBeInTheDocument();
    expect(within(qaRuns).getByText("src/example.py:1-3 · example")).toBeInTheDocument();
    expect(within(qaRuns).getByText("90.0%")).toBeInTheDocument();
    fireEvent.click(within(screen.getByLabelText("Final thesis evaluation section")).getByRole("button", { name: "Search" }));
    expect(screen.getByText("Search by embedding arm")).toBeInTheDocument();
    expect(screen.getAllByText("gemini-embedding-2").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Context strategy experiment" }));
    expect(screen.getByText("Repository context strategy")).toBeInTheDocument();
    expect(screen.getAllByText("+42.5 pp").length).toBeGreaterThan(0);
    expect(screen.getByText("85.8%")).toBeInTheDocument();
    expect(screen.getByText("3 · Does a tool-using Git Agent help?")).toBeInTheDocument();
    expect(screen.getByText(/median latency was 2.34× higher/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Provider settings" }));
    expect(screen.getByRole("dialog", { name: "Provider settings" })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("LLM model preset"), { target: { value: "custom" } });
    const providers = screen.getAllByLabelText("Provider");
    fireEvent.change(providers[1], { target: { value: "openai_compatible" } });
    const key = screen.getByLabelText("API key");
    expect(key).toHaveAttribute("type", "password");
    fireEvent.change(key, { target: { value: "DUMMY_FRONTEND_TEST_KEY" } });
    expect([...stored.values()].join("\n")).not.toContain("DUMMY_FRONTEND_TEST_KEY");
    expect(sessionStorage.setItem).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Clear key" }));
    expect(key).toHaveValue("");
    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(screen.queryByRole("dialog", { name: "Provider settings" })).not.toBeInTheDocument();
  });

  it("restores non-sensitive provider preferences and resets each provider independently", async () => {
    installApi();
    const first = render(<App />);
    await ready();

    fireEvent.click(screen.getByRole("button", { name: "Provider settings" }));
    fireEvent.change(screen.getByLabelText("Embedding model preset"), { target: { value: "custom" } });
    fireEvent.change(screen.getByLabelText("LLM model preset"), { target: { value: "custom" } });
    const models = screen.getAllByLabelText("Model");
    fireEvent.change(models[0], { target: { value: "embed-custom" } });
    fireEvent.change(models[1], { target: { value: "llm-custom" } });
    fireEvent.click(screen.getByLabelText("Close provider settings"));
    fireEvent.click(screen.getByText("Advanced"));
    fireEvent.change(screen.getByLabelText("Answer token budget"), { target: { value: "1024" } });
    await waitFor(() => expect([...stored.values()].join("\n")).toContain("embed-custom"));

    first.unmount();
    render(<App />);
    await ready();
    expect(screen.getByLabelText("Answer token budget")).toHaveValue(1024);
    fireEvent.click(screen.getByRole("button", { name: "Provider settings" }));
    expect(screen.getAllByLabelText("Model")[0]).toHaveValue("embed-custom");
    expect(screen.getAllByLabelText("Model")[1]).toHaveValue("llm-custom");

    fireEvent.click(screen.getAllByRole("button", { name: "Reset to backend defaults" })[1]);
    expect(screen.getByLabelText("Embedding model preset")).toHaveValue("custom");
    expect(screen.getAllByLabelText("Model")[0]).toHaveValue("embed-custom");
    expect(screen.getByLabelText("LLM model preset")).toHaveValue("backend_default");
    expect(screen.getAllByLabelText("Model")[1]).toHaveValue("");
  });

  it("fails closed to backend defaults when saved preferences are corrupt", async () => {
    stored.set("codecompass.preferences.v1", "{not-json");
    installApi();
    render(<App />);
    await ready();
    fireEvent.click(screen.getByRole("button", { name: "Provider settings" }));
    expect(screen.getByLabelText("Embedding model preset")).toHaveValue("backend_default");
    expect(screen.getByLabelText("LLM model preset")).toHaveValue("backend_default");
  });

  it("rejects invalid persisted numeric preferences and restores valid values", () => {
    stored.set("codecompass.preferences.v1", JSON.stringify([]));
    expect(loadPreferences().embedding.useBackendDefault).toBe(true);

    stored.clear();
    stored.set("codecompass.preferences.v0", JSON.stringify({
      embedding: { useBackendDefault: false, provider: "ollama", model: "old-model" },
    }));
    expect(loadPreferences().embedding).not.toHaveProperty("model", "old-model");

    stored.clear();
    stored.set("codecompass.preferences.v1", JSON.stringify({
      embedding: { useBackendDefault: false, provider: "ollama", timeoutSeconds: "not-a-number", dimensions: "Infinity" },
      llm: { useBackendDefault: false, provider: "openai_compatible", timeoutSeconds: "601" },
      answerTokenBudget: "1.5",
    }));

    const invalid = loadPreferences();
    expect(invalid.embedding.timeoutSeconds).toBe("");
    expect(invalid.embedding.dimensions).toBe("");
    expect(invalid.llm.timeoutSeconds).toBe("");
    expect(invalid.answerTokenBudget).toBe("");

    stored.set("codecompass.preferences.v1", JSON.stringify({
      embedding: { useBackendDefault: false, provider: "ollama", timeoutSeconds: "45", dimensions: "768" },
      llm: { useBackendDefault: false, provider: "openai_compatible", timeoutSeconds: "90" },
      answerTokenBudget: "1024",
    }));

    const valid = loadPreferences();
    expect(valid.embedding.timeoutSeconds).toBe("45");
    expect(valid.embedding.dimensions).toBe("768");
    expect(valid.llm.timeoutSeconds).toBe("90");
    expect(valid.answerTokenBudget).toBe("1024");
  });

  it("renders grounded Ask citations and navigates by file_id directly", async () => {
    const fetchMock = installApi((path) => path === "/projects/1/ask" ? response({
      question: "How does escape_silent handle None?",
      answer: "It returns an empty Markup instance.",
      method: "hybrid",
      citations: [citation],
      omitted_context_count: 0,
      llm_model: "qwen",
      llm_provider: "ollama",
    }) : undefined);
    render(<App />);
    await ready();

    fireEvent.change(screen.getByLabelText("Ask about this codebase"), { target: { value: "How does escape_silent handle None?" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask CodeQAPass" }));
    expect(screen.getByRole("status")).toHaveTextContent("Retrieving evidence and generating answer...");
    expect(await screen.findByText("It returns an empty Markup instance.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Edit question" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "New question" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy answer" })).toBeInTheDocument();
    expect(screen.getByText("L48–61")).toBeInTheDocument();
    expect(screen.getByText("escape_silent").closest(".citation-row")).toHaveClass("primary");
    fireEvent.click(screen.getByRole("button", { name: /Open code/i }));
    expect(await screen.findByTestId("monaco")).toHaveTextContent("def escape_silent");
    expect(screen.getByText("Lines 48–61")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: file.relative_path })).toHaveClass("selected");
    expect(fetchMock).toHaveBeenCalledWith("/api/projects/1/files/4/content", expect.anything());
    fireEvent.click(screen.getByRole("button", { name: "Edit question" }));
    expect(screen.getByLabelText("Ask about this codebase")).toHaveValue("How does escape_silent handle None?");
    fireEvent.click(screen.getByRole("button", { name: "Ask CodeQAPass" }));
    expect(await screen.findByRole("button", { name: "New question" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "New question" }));
    expect(screen.getByLabelText("Ask about this codebase")).toHaveValue("");
    expect(screen.queryByText("It returns an empty Markup instance.")).not.toBeInTheDocument();
  });

  it("renders safe grounded Markdown without interpreting raw HTML", async () => {
    installApi((path) => path === "/projects/1/ask" ? response({
      question: "Explain it",
      answer: "**Result**\n\n1. Call `escape_silent`\n2. Return safely\n\n<script>window.pwned = true</script>",
      method: "hybrid",
      citations: [],
      omitted_context_count: 0,
      llm_model: "qwen",
      llm_provider: "ollama",
    }) : undefined);
    const { container } = render(<App />);
    await ready();

    fireEvent.change(screen.getByLabelText("Ask about this codebase"), { target: { value: "Explain it" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask CodeQAPass" }));
    expect((await screen.findByText("Result")).tagName).toBe("STRONG");
    expect(screen.getByText("escape_silent")).toHaveAttribute("dir", "ltr");
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    expect(container.querySelector("script")).toBeNull();
  });

  it("shows truncation only from explicit provider metadata", async () => {
    let truncated = false;
    installApi((path) => path === "/projects/1/ask" ? response({
      question: "Question",
      answer: "An answer that may look unfinished",
      method: "hybrid",
      citations: [],
      omitted_context_count: 0,
      llm_model: "model",
      llm_provider: "provider",
      finish_reason: truncated ? "length" : null,
    }) : undefined);
    render(<App />);
    await ready();
    fireEvent.change(screen.getByLabelText("Ask about this codebase"), { target: { value: "Question" } });

    fireEvent.click(screen.getByRole("button", { name: "Ask CodeQAPass" }));
    await screen.findByText("An answer that may look unfinished");
    expect(screen.queryByText("The answer reached its token limit.")).not.toBeInTheDocument();

    truncated = true;
    fireEvent.click(screen.getByRole("button", { name: "Edit question" }));
    fireEvent.click(screen.getByRole("button", { name: "Ask CodeQAPass" }));
    expect(await screen.findByText("The answer reached its token limit.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Edit question" }));
    expect(screen.getByLabelText("Answer token budget")).toBeVisible();
  });

  it("omits an empty Ask budget, sends a valid override, and blocks invalid values", async () => {
    const askBodies: Record<string, unknown>[] = [];
    installApi((path, init) => {
      if (path !== "/projects/1/ask") return undefined;
      askBodies.push(JSON.parse(String(init?.body)));
      return response({ question: "Question", answer: "Answer", method: "hybrid", citations: [], omitted_context_count: 0, llm_model: "model", llm_provider: "provider" });
    });
    render(<App />);
    await ready();

    fireEvent.change(screen.getByLabelText("Ask about this codebase"), { target: { value: "Question" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask CodeQAPass" }));
    await waitFor(() => expect(askBodies).toHaveLength(1));
    expect(askBodies[0]).not.toHaveProperty("max_tokens");

    fireEvent.click(screen.getByRole("button", { name: "Edit question" }));
    fireEvent.click(screen.getByText("Advanced"));
    const budget = screen.getByLabelText("Answer token budget");
    fireEvent.change(budget, { target: { value: "1024" } });
    fireEvent.click(screen.getByRole("button", { name: "Provider settings" }));
    fireEvent.change(screen.getByLabelText("LLM model preset"), { target: { value: "custom" } });
    fireEvent.change(screen.getAllByLabelText("Model")[1], { target: { value: "another-model" } });
    expect(budget).toHaveValue(1024);
    fireEvent.click(screen.getByRole("button", { name: "Ask CodeQAPass" }));
    await waitFor(() => expect(askBodies).toHaveLength(2));
    expect(askBodies[1]).toHaveProperty("max_tokens", 1024);

    fireEvent.click(screen.getByRole("button", { name: "Edit question" }));
    const updatedBudget = screen.getByLabelText("Answer token budget");
    const askAdvanced = updatedBudget.closest("details");
    if (!askAdvanced?.open) fireEvent.click(within(askAdvanced!).getByText("Advanced"));
    for (const invalid of ["0", "-1", "8001", "1.5"]) {
      fireEvent.change(updatedBudget, { target: { value: invalid } });
      expect(screen.getByRole("alert")).toHaveTextContent("Enter an integer from 1 to 8000.");
      expect(screen.getByRole("button", { name: "Ask CodeQAPass" })).toBeDisabled();
    }
    expect(askBodies).toHaveLength(2);
  });

  it("shows real indexing stages and counters, then refreshes the indexed project", async () => {
    let indexed = false;
    installApi((path, init) => {
      if (path === "/projects") return response(indexed ? [project] : []);
      if (path === "/projects/index-jobs" && init?.method === "POST") return response({
        id: "job-1", state: "scanning", operation: "indexed", project_id: null,
        observed_stages: ["scanning"],
        counters: { files_discovered: 3 }, started_at: "2026-01-01", updated_at: "2026-01-01",
        completed_at: null, elapsed_seconds: 0.4, previous_index_preserved: null, result: null, error: null,
      }, 202);
      if (path === "/projects/index-jobs/job-1") {
        indexed = true;
        return response({
          id: "job-1", state: "completed", operation: "indexed", project_id: 1,
          observed_stages: ["scanning", "parsing", "chunking", "preflight", "embedding", "verifying", "activating"],
          counters: { files_discovered: 3, files_parsed: 3, chunks_generated: 4, embeddings_completed: 4, chunks_expected: 4, vectors_stored: 4 },
          started_at: "2026-01-01", updated_at: "2026-01-01", completed_at: "2026-01-01",
          elapsed_seconds: 1.5, previous_index_preserved: false,
          result: { project_id: 1, operation: "indexed", complete: true, structural_stats: { files_indexed: 1 }, vector_stats: { vectors_stored: 1 }, embedding: { provider: "ollama", model: "embed", dimensions: 768 } },
          error: null,
        });
      }
      return undefined;
    });
    render(<App />);
    expect(await screen.findByText("Connect your first repository")).toBeInTheDocument();
    expect(screen.getByText("No project")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Repository path"), { target: { value: "<temporary-repository>" } });
    fireEvent.click(screen.getByRole("button", { name: "Index repository" }));
    expect(await screen.findByRole("button", { name: "Working..." })).toBeDisabled();
    expect(await screen.findByText("Check source changes")).toBeInTheDocument();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
    expect(screen.getByText("Technical details").closest("details")).not.toHaveAttribute("open");
    expect(screen.getByText(/Files discovered:/)).toHaveTextContent("3");
    expect(screen.queryByText("100%")).not.toBeInTheDocument();
    expect([...stored.values()].join("\n")).not.toContain("<temporary-repository>");

    expect(await screen.findByText("MarkupSafe", {}, { timeout: 2500 })).toBeInTheDocument();
    expect(screen.getByText("Verified")).toBeInTheDocument();
    for (const label of ["Check source changes", "Parse symbols", "Build chunks", "Provider preflight", "Generate embeddings", "Verify vectors", "Activate index"]) {
      expect(screen.getByText(label).closest("li")).toHaveClass("complete");
    }
  });

  it("shows an incremental no-op without invented progress", async () => {
    let polls = 0;
    installApi((path, init) => {
      if (path === "/projects/index-jobs" && init?.method === "POST") return response({
        id: "job-noop", state: "scanning", operation: "reindexed", project_id: 1,
        observed_stages: ["scanning"],
        counters: { files_discovered: 1 }, started_at: "2026-01-01", updated_at: "2026-01-01",
        completed_at: null, elapsed_seconds: 0.1, previous_index_preserved: null, result: null, error: null,
      }, 202);
      if (path === "/projects/index-jobs/job-noop") {
        polls += 1;
        return response({
        id: "job-noop", state: "completed", operation: "reindexed", project_id: 1,
        observed_stages: ["scanning"],
        counters: { files_discovered: 1, files_unchanged: 1, files_added: 0, files_modified: 0, files_deleted: 0, chunks_reused: 1, vectors_reused: 1, vectors_deleted: 0 },
        started_at: "2026-01-01", updated_at: "2026-01-01", completed_at: "2026-01-01",
        elapsed_seconds: 0.4, previous_index_preserved: null,
        result: { project_id: 1, operation: "reindexed", strategy: "incremental", no_changes: true, complete: true, structural_stats: {}, vector_stats: {}, embedding: { provider: "ollama", model: "embed", dimensions: 768 } },
        error: null,
        });
      }
      return undefined;
    });
    render(<App />);
    await ready();
    fireEvent.click(screen.getByRole("button", { name: "Repository" }));
    fireEvent.click(screen.getByRole("button", { name: "Index / Re-index" }));

    expect(await screen.findByText("Repository already up to date", {}, { timeout: 2500 })).toBeInTheDocument();
    expect(screen.getByText("Existing verified index retained")).toBeInTheDocument();
    expect(screen.getByText(/Files unchanged:/)).toHaveTextContent("1");
    const summary = screen.getByLabelText("Indexing counters");
    expect(summary).toHaveTextContent("Files1");
    expect(summary).toHaveTextContent("Symbols1");
    expect(summary).toHaveTextContent("Chunks1");
    expect(screen.queryByRole("progressbar", { name: "Observed indexing stages" })).not.toBeInTheDocument();
    expect(screen.getByText("Check source changes").closest("li")).toHaveClass("complete");
    for (const label of ["Parse symbols", "Build chunks", "Provider preflight", "Generate embeddings", "Verify vectors", "Activate index"]) {
      expect(screen.queryByText(label)).not.toBeInTheDocument();
    }
    await new Promise((resolve) => setTimeout(resolve, 1100));
    expect(polls).toBe(1);
  });

  it("shows only observed stages for a delete-only update", async () => {
    installApi((path, init) => {
      if (path === "/projects/index-jobs" && init?.method === "POST") return response({
        id: "job-delete", state: "scanning", operation: "reindexed", project_id: 1,
        observed_stages: ["scanning"], counters: {}, started_at: "2026-01-01", updated_at: "2026-01-01",
        completed_at: null, elapsed_seconds: 0.1, previous_index_preserved: null, result: null, error: null,
      }, 202);
      if (path === "/projects/index-jobs/job-delete") return response({
        id: "job-delete", state: "completed", operation: "reindexed", project_id: 1,
        observed_stages: ["scanning", "verifying", "activating"], counters: { files_deleted: 1, vectors_deleted: 1 },
        started_at: "2026-01-01", updated_at: "2026-01-01", completed_at: "2026-01-01", elapsed_seconds: 0.4,
        previous_index_preserved: null,
        result: { project_id: 1, operation: "reindexed", strategy: "incremental", no_changes: false, complete: true, structural_stats: {}, vector_stats: {}, embedding: { provider: "ollama", model: "embed", dimensions: 768 } },
        error: null,
      });
      return undefined;
    });
    render(<App />);
    await ready();
    fireEvent.click(screen.getByRole("button", { name: "Repository" }));
    fireEvent.click(screen.getByRole("button", { name: "Index / Re-index" }));

    expect(await screen.findByText("Incremental update complete", {}, { timeout: 2500 })).toBeInTheDocument();
    for (const label of ["Check source changes", "Verify vectors", "Activate index"]) {
      expect(screen.getByText(label).closest("li")).toHaveClass("complete");
    }
    for (const label of ["Parse symbols", "Build chunks", "Provider preflight", "Generate embeddings"]) {
      expect(screen.getByText(label).closest("li")).toHaveClass("pending");
    }
  });

  it("resumes an active indexing job after refresh", async () => {
    let activeReturned = false;
    let retryCalls = 0;
    installApi((path, init) => {
      if (path === "/projects/index-jobs/active" && !activeReturned) {
        activeReturned = true;
        return response({
          id: "job-resume", state: "embedding", operation: "reindexed", project_id: 1,
          observed_stages: ["scanning", "parsing", "chunking", "preflight", "embedding"],
          counters: { chunks_generated: 8, chunks_expected: 8, embeddings_completed: 5 }, started_at: "2026-01-01", updated_at: "2026-01-01",
          completed_at: null, elapsed_seconds: 4.2, previous_index_preserved: null, result: null, error: null,
        });
      }
      if (path === "/projects/index-jobs/job-resume") return response({
        id: "job-resume", state: "failed", operation: "reindexed", project_id: 1,
        observed_stages: ["scanning", "parsing", "chunking", "preflight", "embedding"],
        counters: { chunks_generated: 8, chunks_expected: 8, embeddings_completed: 5 }, started_at: "2026-01-01", updated_at: "2026-01-01",
        completed_at: "2026-01-01", elapsed_seconds: 5.1, previous_index_preserved: true, result: null,
        error: { code: "embedding_provider_unavailable", message: "Embedding provider is unavailable.", stage: "embedding", error_type: "connection" },
      });
      if (path === "/projects/index-jobs" && init?.method === "POST") {
        retryCalls += 1;
        return response({
          id: "job-retry", state: "failed", operation: "reindexed", project_id: 1, counters: {},
          started_at: "2026-01-01", updated_at: "2026-01-01", completed_at: "2026-01-01",
          elapsed_seconds: 0.1, previous_index_preserved: true, result: null,
          error: { code: "embedding_provider_unavailable", message: "Embedding provider is unavailable.", stage: "preflight" },
        }, 202);
      }
      return undefined;
    });
    render(<App />);
    await ready();

    expect(await screen.findByText("Generate embeddings")).toBeInTheDocument();
    expect(screen.getByText("Updating index")).toHaveAttribute("title", "Current verified index remains available");
    expect(screen.getByText(/Generate embeddings · Stage 5 of 7/)).toBeInTheDocument();
    expect(screen.getByText("5 / 8")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Embedding progress" })).toHaveAttribute("aria-valuenow", "5");
    expect(screen.getByRole("progressbar", { name: "Embedding progress" })).toHaveAttribute("aria-valuemax", "8");
    expect(screen.getByRole("progressbar", { name: "Embedding progress" }).querySelector("i")).toHaveStyle({ width: "62.5%" });
    expect(screen.getByText(/Embeddings completed:/)).toHaveTextContent("5");
    expect(await screen.findByText("Previous index remains available.", {}, { timeout: 2500 })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: file.relative_path })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Retry / Re-index" }));
    await waitFor(() => expect(retryCalls).toBe(1));
  });

  it("labels known incomplete vector state and offers re-indexing", async () => {
    const incomplete = { ...project, vector_complete: false };
    installApi((path) => path === "/projects/1" ? response(incomplete) : undefined);
    render(<App />);
    await ready();

    expect(screen.queryByText("Needs attention")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Vector index incomplete" }));
    expect(screen.getByText("Index another repository or refresh the current source")).toBeInTheDocument();
  });

  it("does not claim there is no project when project loading fails", async () => {
    installApi((path) => path === "/projects" ? Promise.reject(new Error("unavailable")) : undefined);
    render(<App />);

    expect(await screen.findByText("Project status unavailable")).toBeInTheDocument();
    expect(screen.queryByText("No project")).not.toBeInTheDocument();
  });

  it("turns source_changed into a safe re-index action", async () => {
    installApi((path) => path === "/projects/1/files/4/content"
      ? response({ error: { code: "source_changed", message: "private detail", details: { path: "PRIVATE_BACKEND_PATH" } } }, 409)
      : undefined);
    render(<App />);
    await ready();
    fireEvent.click(screen.getByRole("button", { name: file.relative_path }));
    expect(await screen.findByText("Source changed")).toBeInTheDocument();
    expect(screen.getByText("Source file changed after indexing. Re-index the repository.")).toBeInTheDocument();
    expect(screen.queryByText("PRIVATE_BACKEND_PATH")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Re-index repository" }));
    expect(screen.getByText("Index another repository or refresh the current source")).toBeInTheDocument();
  });

  it("preserves ordered Search results and explains embedding mismatch safely", async () => {
    let mismatch = false;
    let searchBody: Record<string, unknown> = {};
    installApi((path, init) => {
      if (path !== "/projects/1/search") return undefined;
      searchBody = JSON.parse(String(init?.body));
      if (mismatch) return response({ error: { code: "embedding_configuration_mismatch", message: "raw", details: { internal_path: "PRIVATE_BACKEND_PATH" } } }, 409);
      return response({ query: "escape", method: "lexical", results: [
        { ...citation, score: 4.2, code: "def escape_silent(): pass", retrieval_method: "lexical" },
        { ...citation, chunk_id: "chunk-escape-2", qualified_name: "escape", symbol_name: "escape", score: 2.1, start_line: 24, end_line: 45, code: "def escape(): pass", retrieval_method: "lexical" },
      ] });
    });
    render(<App />);
    await ready();
    fireEvent.click(screen.getByRole("tab", { name: "Search" }));
    const searchQuery = screen.getByLabelText("Search indexed code");
    const searchRow = searchQuery.closest<HTMLElement>(".search-input-row")!;
    expect(within(searchRow).getByRole("button", { name: "Hybrid" })).toBeInTheDocument();
    expect(within(searchRow).getByLabelText("Result limit")).toHaveValue(10);
    fireEvent.change(searchQuery, { target: { value: "escape" } });
    fireEvent.change(screen.getByLabelText("Result limit"), { target: { value: "20" } });
    fireEvent.click(screen.getByRole("button", { name: "Lexical" }));
    fireEvent.click(screen.getByRole("button", { name: /^Search$/ }));
    await screen.findByText("2 results");
    expect(searchBody).toMatchObject({ method: "lexical", limit: 20 });
    const results = screen.getAllByRole("article");
    expect(within(results[0]).getByText("escape_silent")).toBeInTheDocument();
    expect(within(results[1]).getByText("escape")).toBeInTheDocument();

    mismatch = true;
    fireEvent.click(screen.getByRole("button", { name: "Hybrid" }));
    fireEvent.click(screen.getByRole("button", { name: /^Search$/ }));
    expect(await screen.findByText("Embedding configuration mismatch")).toBeInTheDocument();
    expect(screen.queryByText("PRIVATE_BACKEND_PATH")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Re-index repository" })).toBeInTheDocument();
  });

  it("separates extracted/generated documentation and resolves ambiguity explicitly", async () => {
    let calls = 0;
    let documentationRequest: Record<string, unknown> = {};
    installApi((path, init) => {
      if (path !== "/projects/1/documentation") return undefined;
      calls += 1;
      documentationRequest = JSON.parse(String(init?.body));
      if (calls === 1) return response({ error: { code: "documentation_ambiguous", message: "Multiple matches", details: { candidates: [{ symbol_id: 12, chunk_id: "chunk-escape", symbol_type: "function", qualified_name: "escape_silent", relative_source_path: file.relative_path, start_line: 48, end_line: 61 }] } } }, 409);
      return response({
        extracted: { citation: { project_id: 1, project_name: "MarkupSafe", file_id: 4, symbol_id: 12, chunk_id: "chunk-escape", qualified_name: "escape_silent", relative_source_path: file.relative_path, start_line: 48, end_line: 61, content_hash: "content" }, symbol_type: "function", signature: "def escape_silent(s) -> Markup", parameters: ["s"], return_annotation: "Markup", is_async: false, source_file_hash: "abc" },
        generated: { summary: "برای None یک Markup خالی برمی‌گرداند.", detailed_description: "پیش از فراخوانی escape() ورودی را بررسی می‌کند.", parameters: [{ name: "s", description: "مقداری که باید escape شود." }], return_value: "یک Markup امن.", raises: [], side_effects: [], dependencies: ["escape()"], notes: [] },
        citations: [{ project_id: 1, project_name: "MarkupSafe", file_id: 4, symbol_id: 12, chunk_id: "chunk-escape", qualified_name: "escape_silent", relative_source_path: file.relative_path, start_line: 48, end_line: 61, content_hash: "content" }],
        generation: { schema_version: "1", provider: "ollama", model: "qwen", language: "fa" },
      });
    });
    render(<App />);
    await ready();
    fireEvent.click(screen.getByRole("tab", { name: "Documentation" }));
    fireEvent.change(screen.getByLabelText("Function or method"), { target: { value: "escape_silent" } });
    fireEvent.click(screen.getByRole("button", { name: "Persian" }));
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    expect(await screen.findByText("Choose a matching symbol")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /escape_silent.*src\/markupsafe/i }));
    expect(await screen.findByText("Extracted facts")).toBeInTheDocument();
    expect(screen.getByText("Generated explanation")).toBeInTheDocument();
    expect(screen.getByText("def escape_silent(s) -> Markup")).toBeInTheDocument();
    expect(screen.getAllByText("escape()", { selector: "bdi" })).not.toHaveLength(0);
    for (const identifier of screen.getAllByText("escape()", { selector: "bdi" })) {
      expect(identifier).toHaveAttribute("dir", "ltr");
    }
    expect(documentationRequest).toMatchObject({ language: "fa", max_tokens: 2400 });
  });
});
