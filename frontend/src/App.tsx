import { Check, ChevronDown, Compass, Database, FolderCog, Menu, PanelLeftClose, Settings, ShieldCheck } from "lucide-react";
import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";

import { api, embeddingOverride, providerOverride } from "./api/client";
import type {
  AskResponse,
  ContextStrategyEvaluationResponse,
  DocumentationResponse,
  EvaluationResponse,
  FinalThesisEvaluationResponse,
  IndexJob,
  Project,
  RetrievalMethod,
  SearchResponse,
  SourceFile,
  SymbolItem,
} from "./api/types";
import type { SourceSelection } from "./components/CodeDrawer";
import type { NavigationCitation } from "./components/CitationList";
import { ErrorMessage } from "./components/ErrorMessage";
import { EvaluationPanel } from "./components/EvaluationPanel";
import { ProjectExplorer } from "./components/ProjectExplorer";
import { ProjectSetup } from "./components/ProjectSetup";
import { ProviderSettings } from "./components/ProviderSettings";
import { Workspace, type WorkspaceTab } from "./components/Workspace";
import { defaultEmbedding, defaultLlm, loadPreferences, savePreferences } from "./preferences";

const CodeDrawer = lazy(() => import("./components/CodeDrawer").then((module) => ({ default: module.CodeDrawer })));

interface RequestState<T> {
  result: T | null;
  loading: boolean;
  error: unknown;
}

const idle = <T,>(): RequestState<T> => ({ result: null, loading: false, error: null });

function ProjectSwitcher({ projects, project, loading, onSelect }: {
  projects: Project[];
  project: Project | null;
  loading: boolean;
  onSelect: (projectId: number) => void;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent | KeyboardEvent) => {
      if (event.type === "keydown" && (event as KeyboardEvent).key !== "Escape") return;
      if (event.type === "pointerdown" && root.current?.contains(event.target as Node)) return;
      setOpen(false);
    };
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("pointerdown", close);
      document.removeEventListener("keydown", close);
    };
  }, [open]);

  return (
    <div className={`project-switcher${open ? " open" : ""}`} ref={root}>
      <Database size={16} aria-hidden="true" />
      <button
        className="project-switcher-trigger"
        type="button"
        aria-label={`Current project: ${project?.name ?? "No indexed projects"}`}
        aria-haspopup="menu"
        aria-expanded={open}
        disabled={!projects.length || loading}
        onClick={() => setOpen((value) => !value)}
      >
        <span>{project?.name ?? "No indexed projects"}</span><ChevronDown size={15} />
      </button>
      {open ? (
        <div className="project-menu" role="menu" aria-label="Projects">
          <span className="project-menu-label">Indexed repositories</span>
          {projects.map((item) => (
            <button
              type="button"
              role="menuitemradio"
              aria-checked={item.id === project?.id}
              key={item.id}
              onClick={() => { setOpen(false); if (item.id !== project?.id) onSelect(item.id); }}
            >
              <strong>{item.name}</strong>
              {item.id === project?.id ? <Check size={16} aria-hidden="true" /> : null}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export default function App() {
  const [savedPreferences] = useState(loadPreferences);
  const [projects, setProjects] = useState<Project[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [files, setFiles] = useState<SourceFile[]>([]);
  const [symbols, setSymbols] = useState<SymbolItem[]>([]);
  const [repositoryPath, setRepositoryPath] = useState("");
  const [projectLoading, setProjectLoading] = useState(true);
  const [projectError, setProjectError] = useState<unknown>(null);
  const [indexJob, setIndexJob] = useState<IndexJob | null>(null);
  const [indexError, setIndexError] = useState<unknown>(null);
  const [embedding, setEmbedding] = useState(savedPreferences.embedding);
  const [llm, setLlm] = useState(savedPreferences.llm);
  const [answerTokenBudget, setAnswerTokenBudget] = useState(savedPreferences.answerTokenBudget);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [setupOpen, setSetupOpen] = useState(false);
  const [explorerOpen, setExplorerOpen] = useState(() => window.innerWidth > 900);
  const [workspaceTab, setWorkspaceTab] = useState<WorkspaceTab>("ask");
  const [askState, setAskState] = useState<RequestState<AskResponse>>(idle);
  const [searchState, setSearchState] = useState<RequestState<SearchResponse>>(idle);
  const [documentationState, setDocumentationState] = useState<RequestState<DocumentationResponse>>(idle);
  const [documentationPreset, setDocumentationPreset] = useState<string | number | null>(null);
  const [source, setSource] = useState<SourceSelection | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [sourceError, setSourceError] = useState<unknown>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [performance, setPerformance] = useState<EvaluationResponse | null>(null);
  const [finalThesisEvaluation, setFinalThesisEvaluation] = useState<FinalThesisEvaluationResponse | null>(null);
  const [contextStrategyEvaluation, setContextStrategyEvaluation] = useState<ContextStrategyEvaluationResponse | null>(null);
  const [evaluationLoading, setEvaluationLoading] = useState(true);
  const [evaluationError, setEvaluationError] = useState<unknown>(null);

  const resetWorkspace = useCallback(() => {
    setAskState(idle());
    setSearchState(idle());
    setDocumentationState(idle());
    setDocumentationPreset(null);
    setSource(null);
    setSourceError(null);
  }, []);

  const loadProject = useCallback(async (projectId: number) => {
    setProjectLoading(true);
    setProjectError(null);
    resetWorkspace();
    try {
      const [detail, projectFiles, projectSymbols] = await Promise.all([
        api.project(projectId),
        api.files(projectId),
        api.symbols(projectId),
      ]);
      setProject(detail);
      setFiles(projectFiles);
      setSymbols(projectSymbols);
    } catch (error) {
      setProjectError(error);
      setProject(null);
      setFiles([]);
      setSymbols([]);
    } finally {
      setProjectLoading(false);
    }
  }, [resetWorkspace]);

  useEffect(() => {
    let active = true;
    void api.projects().then((items) => {
      if (!active) return;
      setProjects(items);
      setSetupOpen(items.length === 0);
      if (items[0]) void loadProject(items[0].id);
      else setProjectLoading(false);
    }).catch((error) => {
      if (active) { setProjectError(error); setProjectLoading(false); }
    });
    void Promise.all([api.evaluation(), api.performance(), api.finalThesisEvaluation(), api.contextStrategyEvaluation()]).then(([summary, measured, thesis, contextStrategy]) => {
      if (active) { setEvaluation(summary); setPerformance(measured); setFinalThesisEvaluation(thesis); setContextStrategyEvaluation(contextStrategy); }
    }).catch((error) => { if (active) setEvaluationError(error); }).finally(() => { if (active) setEvaluationLoading(false); });
    return () => { active = false; };
  }, [loadProject]);

  useEffect(() => {
    let active = true;
    void api.activeIndexJob().then((job) => {
      if (active && job) {
        setIndexJob(job);
        setSetupOpen(true);
      }
    }).catch((error) => {
      if (active) {
        setIndexError(error);
        setSetupOpen(true);
      }
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    savePreferences({ embedding, llm, answerTokenBudget });
  }, [embedding, llm, answerTokenBudget]);

  const refreshProjects = useCallback(async (projectId: number) => {
    const items = await api.projects();
    setProjects(items);
    await loadProject(projectId);
  }, [loadProject]);

  useEffect(() => {
    if (!indexJob || indexJob.state === "completed" || indexJob.state === "failed") return;
    let active = true;
    const timer = window.setTimeout(() => {
      void api.indexJob(indexJob.id).then(async (next) => {
        if (!active) return;
        setIndexError(null);
        setIndexJob(next);
        if (next.state === "completed" && next.project_id !== null) await refreshProjects(next.project_id);
      }).catch((error) => {
        if (active) {
          setIndexError(error);
          setIndexJob((current) => current ? { ...current } : current);
        }
      });
    }, 1000);
    return () => { active = false; window.clearTimeout(timer); };
  }, [indexJob, refreshProjects]);

  const indexRepository = async () => {
    const repositoryPathValue = repositoryPath.trim();
    if (!repositoryPathValue && !project) return;
    setIndexError(null);
    try {
      const job = await api.startIndexJob(
        repositoryPathValue ? { repositoryPath: repositoryPathValue } : { projectId: project!.id },
        embeddingOverride(embedding),
      );
      setIndexJob(job);
    } catch (error) {
      setIndexError(error);
    }
  };

  const reindex = () => setSetupOpen(true);

  const openSource = async (fileId: number, startLine?: number, endLine?: number) => {
    if (!project) return;
    setSource(null);
    setSourceError(null);
    setSourceLoading(true);
    try {
      const content = await api.source(project.id, fileId);
      setSource({ content, startLine, endLine });
    } catch (error) {
      setSourceError(error);
    } finally {
      setSourceLoading(false);
    }
  };

  const openCitation = (citation: NavigationCitation) => void openSource(citation.fileId, citation.startLine, citation.endLine);

  const ask = async (question: string, method: RetrievalMethod, maxTokens?: number) => {
    if (!project) return;
    setAskState({ result: null, loading: true, error: null });
    try {
      const result = await api.ask(project.id, question, method, embeddingOverride(embedding), providerOverride(llm), maxTokens);
      setAskState({ result, loading: false, error: null });
    } catch (error) {
      setAskState({ result: null, loading: false, error });
    }
  };

  const search = async (query: string, method: RetrievalMethod, limit: number) => {
    if (!project) return;
    setSearchState({ result: null, loading: true, error: null });
    try {
      const result = await api.search(project.id, query, method, limit, embeddingOverride(embedding));
      setSearchState({ result, loading: false, error: null });
    } catch (error) {
      setSearchState({ result: null, loading: false, error });
    }
  };

  const documentSymbol = async (identifier: string | number, language: "en" | "fa") => {
    if (!project) return;
    setDocumentationPreset(identifier);
    setDocumentationState({ result: null, loading: true, error: null });
    try {
      const result = await api.documentation(project.id, identifier, language, providerOverride(llm));
      setDocumentationState({ result, loading: false, error: null });
    } catch (error) {
      setDocumentationState({ result: null, loading: false, error });
    }
  };

  const chooseDocumentation = (symbol: SymbolItem) => {
    setDocumentationPreset(symbol.id);
    setWorkspaceTab("documentation");
  };

  const indexing = Boolean(indexJob && indexJob.state !== "completed" && indexJob.state !== "failed");
  const workspaceHasResult = Boolean(askState.result || searchState.result || documentationState.result);
  const projectStatus = indexing && project ? "Updating index" : indexing ? "Indexing" : projectLoading ? "Loading" : projectError ? "Project status unavailable" : project?.vector_complete ? "Ready" : project ? "Vector index incomplete" : "No project";

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#workspace" aria-label="CodeQAPass workspace">
          <Compass size={25} />
          <span><strong>CodeQAPass</strong><small>Local code intelligence</small></span>
        </a>
        <ProjectSwitcher projects={projects} project={project} loading={projectLoading} onSelect={(projectId) => void loadProject(projectId)} />
        {project && !projectLoading && !project.vector_complete ? (
          <button className="status-badge status-action" type="button" onClick={reindex} title="Re-index repository"><i />{projectStatus}</button>
        ) : (
          <span className={`status-badge ${projectStatus === "Ready" ? "ready" : indexing ? "updating" : "idle"}`} title={indexing && project?.vector_complete ? "Current verified index remains available" : undefined}><i />{projectStatus}</span>
        )}
        <div className="top-actions">
          <button className="secondary-button mobile-explorer-toggle" type="button" onClick={() => setExplorerOpen((current) => !current)}><Menu size={17} /> Explorer</button>
          <button className="secondary-button" type="button" onClick={() => setSetupOpen((current) => !current)}><FolderCog size={17} /> Repository</button>
          <button className="secondary-button" type="button" aria-label="Provider settings" onClick={() => setSettingsOpen(true)}><Settings size={17} /> Settings</button>
        </div>
      </header>

      {setupOpen ? <ProjectSetup path={repositoryPath} setPath={setRepositoryPath} currentProject={project} job={indexJob} error={indexError} onSubmit={indexRepository} /> : null}
      {projectError ? <div className="global-error"><ErrorMessage error={projectError} onReindex={reindex} /></div> : null}

      <div className={`main-layout ${source || sourceLoading || sourceError ? "with-code" : ""} ${explorerOpen ? "" : "explorer-closed"} ${workspaceHasResult ? "has-results" : ""} ${setupOpen ? "setup-visible" : ""}`} id="workspace">
        {project && explorerOpen ? <ProjectExplorer files={files} symbols={symbols} selectedFileId={source?.content.id ?? null} onOpenFile={(file) => void openSource(file.id)} onOpenSymbol={(symbol) => void openSource(symbol.file_id, symbol.start_line, symbol.end_line)} onDocumentSymbol={chooseDocumentation} /> : null}
        {project ? (
          <Workspace
            tab={workspaceTab}
            setTab={setWorkspaceTab}
            ask={askState}
            search={searchState}
            documentation={{ ...documentationState, preset: documentationPreset }}
            onAsk={ask}
            onNewQuestion={() => setAskState(idle())}
            onSearch={search}
            onDocument={documentSymbol}
            onOpenCitation={openCitation}
            onReindex={reindex}
            answerTokenBudget={answerTokenBudget}
            onAnswerTokenBudgetChange={setAnswerTokenBudget}
          />
        ) : !projectLoading ? (
          <main className="no-project">
            <Compass size={38} />
            <h1>Start with a local Python repository</h1>
            <button className="primary-button" type="button" onClick={() => setSetupOpen(true)}><FolderCog size={17} /> Open repository setup</button>
          </main>
        ) : <main className="no-project"><span className="spinner-large" /><p>Loading workspace...</p></main>}
        {source || sourceLoading || sourceError ? (
          <Suspense fallback={<aside className="code-drawer"><div className="drawer-state">Loading code viewer...</div></aside>}>
            <CodeDrawer selection={source} loading={sourceLoading} error={sourceError} onClose={() => { setSource(null); setSourceError(null); }} onReindex={reindex} />
          </Suspense>
        ) : null}
      </div>

      {project ? (
        <section className="project-facts" aria-label="Repository status">
          <span><strong>{project.files ?? "—"}</strong> Files</span>
          <span><strong>{project.symbols ?? "—"}</strong> Symbols</span>
          <span><strong>{project.chunks ?? "—"}</strong> Chunks</span>
          <span><ShieldCheck size={16} /><strong>{project.vector_complete ? "Verified" : "Incomplete"}</strong> Vector index</span>
          {!explorerOpen ? <button className="icon-button" type="button" onClick={() => setExplorerOpen(true)} title="Open explorer" aria-label="Open explorer"><PanelLeftClose size={17} /></button> : null}
        </section>
      ) : null}

      <EvaluationPanel summary={evaluation} performance={performance} finalThesis={finalThesisEvaluation} contextStrategy={contextStrategyEvaluation} loading={evaluationLoading} error={evaluationError} />
      <ProviderSettings
        open={settingsOpen}
        embedding={embedding}
        llm={llm}
        onEmbeddingChange={setEmbedding}
        onLlmChange={setLlm}
        onResetEmbedding={() => setEmbedding(defaultEmbedding)}
        onResetLlm={() => setLlm(defaultLlm)}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  );
}
