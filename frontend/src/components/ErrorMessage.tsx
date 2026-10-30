import { AlertCircle, RefreshCw } from "lucide-react";

import { ApiError } from "../api/client";

const messages: Record<string, { title: string; message: string; action?: string }> = {
  embedding_configuration_mismatch: {
    title: "Embedding configuration mismatch",
    message: "This repository was indexed with a different embedding configuration. Re-index it before using Semantic or Hybrid search.",
    action: "Re-index repository",
  },
  vector_index_state_invalid: {
    title: "Vector index unavailable",
    message: "The vector index state is invalid or incomplete. Re-index the repository to repair it.",
    action: "Re-index repository",
  },
  source_changed: {
    title: "Source changed",
    message: "Source file changed after indexing. Re-index the repository.",
    action: "Re-index repository",
  },
  documentation_not_found: {
    title: "Symbol not found",
    message: "No matching indexed symbol was found.",
  },
  documentation_provider_timeout: {
    title: "Provider timed out",
    message: "The model provider did not respond in time. Check provider settings and retry.",
  },
  provider_configuration_invalid: {
    title: "Provider configuration invalid",
    message: "Check the provider, base URL, model, and API key settings.",
  },
  retrieval_embedding_failed: {
    title: "Embedding provider failed",
    message: "The embedding provider could not complete the request. Check provider settings and retry.",
  },
  qa_llm_failed: {
    title: "Answer generation failed",
    message: "The LLM provider could not generate an answer. Check provider settings and retry.",
  },
  validation_error: {
    title: "Check the request",
    message: "One or more values are invalid. Review the highlighted form fields.",
  },
};

export function describeError(error: unknown) {
  if (error instanceof ApiError) {
    return messages[error.code] ?? {
      title: "Request failed",
      message: error.status >= 500 ? "CodeQAPass could not complete the request." : error.message,
    };
  }
  return { title: "Connection failed", message: "The CodeQAPass API is not reachable." };
}

export function ErrorMessage({ error, onReindex }: { error: unknown; onReindex?: () => void }) {
  const detail = describeError(error);
  return (
    <div className="error-message" role="alert">
      <AlertCircle size={18} aria-hidden="true" />
      <div>
        <strong>{detail.title}</strong>
        <p>{detail.message}</p>
      </div>
      {detail.action && onReindex ? (
        <button className="text-button" type="button" onClick={onReindex}>
          <RefreshCw size={15} aria-hidden="true" />
          {detail.action}
        </button>
      ) : null}
    </div>
  );
}
