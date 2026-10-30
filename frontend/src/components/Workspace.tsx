import { AlertTriangle, BookOpen, Braces, Check, Copy, LoaderCircle, MessageSquareText, Pencil, Plus, Search, Send, ShieldCheck, SlidersHorizontal, Sparkles } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";
import ReactMarkdown from "react-markdown";

import { ApiError } from "../api/client";
import type { AskResponse, DocumentationResponse, ResolutionCandidate, RetrievalMethod, SearchResponse } from "../api/types";
import { CitationList, type NavigationCitation } from "./CitationList";
import { ErrorMessage } from "./ErrorMessage";

export type WorkspaceTab = "ask" | "search" | "documentation";

function GroundedMarkdown({ children }: { children: string }) {
  const persian = /[\u0600-\u06ff]/.test(children);
  return (
    <div className={`answer-markdown${persian ? " persian" : ""}`} dir={persian ? "rtl" : "ltr"}>
      <ReactMarkdown
        components={{
          code: ({ children: code }) => <code dir="ltr">{code}</code>,
          pre: ({ children: code }) => <pre dir="ltr">{code}</pre>,
          a: ({ children: label, href }) => <a href={href} target="_blank" rel="noreferrer">{label}</a>,
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

function MethodControl({ value, onChange }: { value: RetrievalMethod; onChange: (value: RetrievalMethod) => void }) {
  return (
    <div className="segmented" aria-label="Retrieval method">
      {(["lexical", "semantic", "hybrid"] as const).map((method) => (
        <button key={method} type="button" className={value === method ? "active" : ""} onClick={() => onChange(method)} aria-pressed={value === method}>
          {method[0].toUpperCase() + method.slice(1)}
        </button>
      ))}
    </div>
  );
}

function AskPanel({ result, loading, error, onAsk, onNewQuestion, onOpenCitation, onReindex, maxTokens, onMaxTokensChange }: {
  result: AskResponse | null;
  loading: boolean;
  error: unknown;
  onAsk: (question: string, method: RetrievalMethod, maxTokens?: number) => void;
  onNewQuestion: () => void;
  onOpenCitation: (citation: NavigationCitation) => void;
  onReindex: () => void;
  maxTokens: string;
  onMaxTokensChange: (value: string) => void;
}) {
  const [question, setQuestion] = useState("");
  const [method, setMethod] = useState<RetrievalMethod>("hybrid");
  const [editing, setEditing] = useState(false);
  const [copied, setCopied] = useState(false);
  const parsedMaxTokens = maxTokens === "" ? undefined : Number(maxTokens);
  const maxTokensError = parsedMaxTokens !== undefined && (!Number.isInteger(parsedMaxTokens) || parsedMaxTokens < 1 || parsedMaxTokens > 8000)
    ? "Enter an integer from 1 to 8000."
    : null;
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (question.trim() && !maxTokensError) {
      setEditing(false);
      setCopied(false);
      onAsk(question.trim(), method, parsedMaxTokens);
    }
  };
  const citations = result?.citations.map((item) => ({
    fileId: item.file_id,
    chunkId: item.chunk_id,
    qualifiedName: item.qualified_name,
    relativePath: item.source_file,
    startLine: item.start_line,
    endLine: item.end_line,
  })) ?? [];
  return (
    <div className={`workspace-panel${!result && !loading && !error ? " empty" : ""}`}>
      {result && !editing ? (
        <div className="compact-question">
          <span dir="auto">{result.question}</span>
          <div className="compact-question-actions">
            <button className="secondary-button" type="button" title="Edit the previous question" onClick={() => { setQuestion(result.question); setEditing(true); }}><Pencil size={15} /> Edit question</button>
            <button className="primary-button" type="button" title="Start a new question" onClick={() => { setQuestion(""); setEditing(true); onNewQuestion(); }}><Plus size={15} /> New question</button>
          </div>
        </div>
      ) : <form className="prompt-form" onSubmit={submit}>
        <label htmlFor="question">Ask about this codebase</label>
        <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="How does escape_silent handle None?" dir="auto" rows={2} />
        <div className="form-actions">
          <div className="ask-controls">
            <MethodControl value={method} onChange={setMethod} />
            <details className="ask-advanced" open={result?.finish_reason === "length" || undefined}>
              <summary><SlidersHorizontal size={14} /> Advanced</summary>
              <div className="advanced-popover">
                <label htmlFor="answer-token-budget">Answer token budget</label>
                <input
                  id="answer-token-budget"
                  type="number"
                  min="1"
                  max="8000"
                  step="1"
                  value={maxTokens}
                  onChange={(event) => onMaxTokensChange(event.target.value)}
                  placeholder="Backend default (180)"
                  aria-invalid={Boolean(maxTokensError)}
                  aria-describedby={maxTokensError ? "answer-token-budget-error" : undefined}
                />
                <small>Overrides the backend default for this answer.</small>
                {maxTokensError ? <p id="answer-token-budget-error" className="validation-message" role="alert">{maxTokensError}</p> : null}
              </div>
            </details>
          </div>
          <button className="primary-button" type="submit" disabled={loading || !question.trim() || Boolean(maxTokensError)}>
            {loading ? <LoaderCircle className="spin" size={17} /> : <Send size={17} />}
            {loading ? "Generating answer..." : "Ask CodeQAPass"}
          </button>
        </div>
      </form>}
      {error ? <ErrorMessage error={error} onReindex={onReindex} /> : null}
      {loading ? (
        <section className="answer-loading" role="status" aria-live="polite">
          <div><LoaderCircle className="spin" size={18} /><strong>Retrieving evidence and generating answer...</strong></div>
          <span /><span /><span />
        </section>
      ) : result ? (
        <section className="answer-section" aria-live="polite">
          <div className="answer-heading">
            <Sparkles size={18} /><span>Grounded answer</span>
            <div className="answer-toolbar">
              <span className="answer-badge">{result.method}</span>
              <span className="answer-badge model" dir="ltr" title={result.llm_model ?? "backend model"}>{result.llm_model ?? "backend model"}</span>
              {result.finish_reason ? <span className={`answer-badge ${result.finish_reason === "length" ? "warning" : "success"}`}>{result.finish_reason === "length" ? "Token limit" : result.finish_reason === "stop" ? "Complete" : result.finish_reason}</span> : null}
              <button className="icon-button" type="button" title="Copy answer" aria-label="Copy answer" onClick={() => void navigator.clipboard?.writeText(result.answer).then(() => setCopied(true)).catch(() => setCopied(false))}>
                {copied ? <Check size={16} /> : <Copy size={16} />}
                <span className="copy-label">{copied ? "Copied" : "Copy"}</span>
              </button>
            </div>
          </div>
          {result.finish_reason === "length" ? <div className="answer-warning" role="status"><AlertTriangle size={17} /><span><strong>The answer reached its token limit.</strong> Increase Answer token budget in Advanced and ask again.</span></div> : null}
          <GroundedMarkdown>{result.answer}</GroundedMarkdown>
          <CitationList citations={citations} onOpen={onOpenCitation} />
        </section>
      ) : !error ? <div className="workspace-empty"><MessageSquareText size={28} /><p>Ask a Persian or English question about the selected project.</p></div> : null}
    </div>
  );
}

function SearchPanel({ result, loading, error, onSearch, onOpenCitation, onReindex }: {
  result: SearchResponse | null;
  loading: boolean;
  error: unknown;
  onSearch: (query: string, method: RetrievalMethod, limit: number) => void;
  onOpenCitation: (citation: NavigationCitation) => void;
  onReindex: () => void;
}) {
  const [query, setQuery] = useState("");
  const [method, setMethod] = useState<RetrievalMethod>("hybrid");
  const [limit, setLimit] = useState(10);
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (query.trim()) onSearch(query.trim(), method, limit);
  };
  return (
    <div className={`workspace-panel${!result && !loading && !error ? " empty" : ""}`}>
      <form className="search-form" onSubmit={submit}>
        <label htmlFor="search-query">Search indexed code</label>
        <div className="search-input-row">
          <div className="search-box"><Search size={17} /><input id="search-query" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="escape HTML input" dir="auto" /></div>
          <MethodControl value={method} onChange={setMethod} />
          <label className="limit-input"><span>Results</span><input aria-label="Result limit" type="number" min="1" max="50" value={limit} onChange={(event) => setLimit(Math.max(1, Math.min(50, Number(event.target.value))))} /></label>
          <button className="primary-button" type="submit" disabled={loading || !query.trim()}>{loading ? <LoaderCircle className="spin" size={17} /> : <Search size={17} />} Search</button>
        </div>
      </form>
      {error ? <ErrorMessage error={error} onReindex={onReindex} /> : null}
      {result ? (
        <section className="search-results" aria-live="polite">
          <div className="result-summary"><strong>{result.results.length} results</strong><span>{result.method}</span></div>
          {result.results.length ? result.results.map((item, index) => (
            <article className="search-result" key={item.chunk_id}>
              <span className="rank">{index + 1}</span>
              <div>
                <h3>{item.qualified_name ?? item.symbol_name ?? "Source chunk"}</h3>
                <p>{item.source_file} · Lines {item.start_line}–{item.end_line}</p>
              </div>
              <span className="score" title="Method-native score">{item.score.toFixed(4)}</span>
              <button className="text-button" type="button" onClick={() => onOpenCitation({ fileId: item.file_id, chunkId: item.chunk_id, qualifiedName: item.qualified_name, relativePath: item.source_file, startLine: item.start_line, endLine: item.end_line })}>Open code</button>
            </article>
          )) : <p className="empty-compact">No matching evidence was found.</p>}
        </section>
      ) : !loading && !error ? <div className="workspace-empty"><Search size={28} /><p>Search with the frozen lexical, semantic, or hybrid configuration.</p></div> : null}
    </div>
  );
}

const TECHNICAL_TOKEN = /(`[^`]+`|[A-Za-z_][A-Za-z0-9_.]*(?:\([^()]*\))?)/g;

function BidiText({ text, isolate }: { text: string; isolate: boolean }) {
  if (!isolate) return text;
  return <>{text.split(TECHNICAL_TOKEN).map((part, index) =>
    /^`[^`]+`$|^[A-Za-z_]/.test(part)
      ? <bdi className="inline-identifier" dir="ltr" key={`${part}-${index}`}>{part.replace(/^`|`$/g, "")}</bdi>
      : part
  )}</>;
}

function ListField({ title, values, isolate }: { title: string; values: string[]; isolate: boolean }) {
  if (!values.length) return null;
  return <div className="doc-field"><dt>{title}</dt><dd><ul>{values.map((value) => <li key={value}><BidiText text={value} isolate={isolate} /></li>)}</ul></dd></div>;
}

function DocumentationPanel({ preset, result, loading, error, onDocument, onOpenCitation }: {
  preset: string | number | null;
  result: DocumentationResponse | null;
  loading: boolean;
  error: unknown;
  onDocument: (identifier: string | number, language: "en" | "fa") => void;
  onOpenCitation: (citation: NavigationCitation) => void;
}) {
  const [identifier, setIdentifier] = useState<string | number>(preset ?? "");
  const [language, setLanguage] = useState<"en" | "fa">("en");
  useEffect(() => { if (preset !== null) setIdentifier(preset); }, [preset]);
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (String(identifier).trim()) onDocument(identifier, language);
  };
  const candidates = error instanceof ApiError && error.code === "documentation_ambiguous"
    ? (error.details.candidates as ResolutionCandidate[] | undefined) ?? []
    : [];
  const citation = result?.citations[0];
  const persian = result?.generation.language === "fa";
  return (
    <div className={`workspace-panel${!result && !loading && !error ? " empty" : ""}`}>
      <form className="documentation-form" onSubmit={submit}>
        <label htmlFor="symbol-identifier">Function or method</label>
        <div className="search-input-row">
          <div className="search-box"><Braces size={17} /><input id="symbol-identifier" value={identifier} onChange={(event) => setIdentifier(event.target.value)} placeholder="escape_silent" spellCheck={false} /></div>
          <div className="segmented language-control" aria-label="Documentation language">
            <button type="button" className={language === "en" ? "active" : ""} onClick={() => setLanguage("en")}>English</button>
            <button type="button" className={language === "fa" ? "active" : ""} onClick={() => setLanguage("fa")}>Persian</button>
          </div>
          <button className="primary-button" type="submit" disabled={loading || !String(identifier).trim()}>{loading ? <LoaderCircle className="spin" size={17} /> : <BookOpen size={17} />} Generate</button>
        </div>
      </form>
      {error ? <ErrorMessage error={error} /> : null}
      {candidates.length ? (
        <div className="candidate-list">
          <h3>Choose a matching symbol</h3>
          {candidates.map((candidate) => (
            <button type="button" key={candidate.symbol_id} onClick={() => { setIdentifier(candidate.symbol_id); onDocument(candidate.symbol_id, language); }}>
              <Braces size={15} /><span><strong>{candidate.qualified_name}</strong><small>{candidate.relative_source_path} · L{candidate.start_line}–{candidate.end_line}</small></span>
            </button>
          ))}
        </div>
      ) : null}
      {result ? (
        <section className="documentation-result" aria-live="polite" dir={result.generation.language === "fa" ? "rtl" : "ltr"}>
          <header>
            <div className="documentation-identity" dir="ltr"><span className="eyebrow">Extracted identity</span><h2>{result.extracted.citation.qualified_name}</h2></div>
            {citation ? <button className="text-button" type="button" onClick={() => onOpenCitation({ fileId: citation.file_id, chunkId: citation.chunk_id, qualifiedName: citation.qualified_name, relativePath: citation.relative_source_path, startLine: citation.start_line, endLine: citation.end_line })}>Open code</button> : null}
          </header>
          <div className="documentation-columns">
            <section className="extracted-facts" dir="ltr">
              <div className="documentation-section-heading"><h3><ShieldCheck size={17} /> Trusted facts</h3><span>Extracted facts</span></div>
              <dl>
                <div><dt>Type</dt><dd>{result.extracted.symbol_type}</dd></div>
                <div><dt>Signature</dt><dd><code>{result.extracted.signature}</code></dd></div>
                <div><dt>Parameters</dt><dd>{result.extracted.parameters.join(", ") || "None"}</dd></div>
                <div><dt>Returns</dt><dd>{result.extracted.return_annotation ?? "Not annotated"}</dd></div>
                <div><dt>Source</dt><dd>{result.extracted.citation.relative_source_path}<br />L{result.extracted.citation.start_line}–{result.extracted.citation.end_line}</dd></div>
              </dl>
            </section>
            <section className="generated-docs" dir={persian ? "rtl" : "ltr"}>
              <div className="documentation-section-heading"><h3><Sparkles size={17} /> Generated explanation</h3><span className="model-badge" dir="ltr">{result.generation.model}</span></div>
              <p className="doc-summary"><BidiText text={result.generated.summary} isolate={persian} /></p>
              <p><BidiText text={result.generated.detailed_description} isolate={persian} /></p>
              <dl>
                {result.generated.parameters.map((parameter) => <div className="doc-field" key={parameter.name}><dt><bdi dir="ltr">{parameter.name}</bdi></dt><dd><BidiText text={parameter.description ?? (persian ? "توضیح بیشتری موجود نیست." : "No additional description.")} isolate={persian} /></dd></div>)}
                {result.generated.return_value ? <div className="doc-field"><dt>Return value</dt><dd><BidiText text={result.generated.return_value} isolate={persian} /></dd></div> : null}
                <ListField title="Raises" values={result.generated.raises} isolate={persian} />
                <ListField title="Side effects" values={result.generated.side_effects} isolate={persian} />
                <ListField title="Dependencies" values={result.generated.dependencies} isolate={persian} />
                <ListField title="Notes" values={result.generated.notes} isolate={persian} />
              </dl>
            </section>
          </div>
        </section>
      ) : !loading && !error ? <div className="workspace-empty"><BookOpen size={28} /><p>Generate source-grounded documentation for an indexed function or method.</p></div> : null}
    </div>
  );
}

export function Workspace(props: {
  tab: WorkspaceTab;
  setTab: (tab: WorkspaceTab) => void;
  ask: { result: AskResponse | null; loading: boolean; error: unknown };
  search: { result: SearchResponse | null; loading: boolean; error: unknown };
  documentation: { result: DocumentationResponse | null; loading: boolean; error: unknown; preset: string | number | null };
  onAsk: (question: string, method: RetrievalMethod, maxTokens?: number) => void;
  onNewQuestion: () => void;
  onSearch: (query: string, method: RetrievalMethod, limit: number) => void;
  onDocument: (identifier: string | number, language: "en" | "fa") => void;
  onOpenCitation: (citation: NavigationCitation) => void;
  onReindex: () => void;
  answerTokenBudget: string;
  onAnswerTokenBudgetChange: (value: string) => void;
}) {
  return (
    <main className="workspace">
      <div className="tab-list workspace-tabs" role="tablist" aria-label="Workspace">
        <button type="button" role="tab" aria-selected={props.tab === "ask"} className={props.tab === "ask" ? "active" : ""} onClick={() => props.setTab("ask")}><MessageSquareText size={17} /> Ask</button>
        <button type="button" role="tab" aria-selected={props.tab === "search"} className={props.tab === "search" ? "active" : ""} onClick={() => props.setTab("search")}><Search size={17} /> Search</button>
        <button type="button" role="tab" aria-selected={props.tab === "documentation"} className={props.tab === "documentation" ? "active" : ""} onClick={() => props.setTab("documentation")}><BookOpen size={17} /> Documentation</button>
      </div>
      {props.tab === "ask" ? <AskPanel {...props.ask} onAsk={props.onAsk} onNewQuestion={props.onNewQuestion} onOpenCitation={props.onOpenCitation} onReindex={props.onReindex} maxTokens={props.answerTokenBudget} onMaxTokensChange={props.onAnswerTokenBudgetChange} /> : null}
      {props.tab === "search" ? <SearchPanel {...props.search} onSearch={props.onSearch} onOpenCitation={props.onOpenCitation} onReindex={props.onReindex} /> : null}
      {props.tab === "documentation" ? <DocumentationPanel {...props.documentation} onDocument={props.onDocument} onOpenCitation={props.onOpenCitation} /> : null}
    </main>
  );
}
