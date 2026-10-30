from codecompass.demo import DEMO_RESPONSE_INSTRUCTIONS, NO_DISPLAYABLE_ANSWER, _DemoPromptBuilder, format_answer
from codecompass.qa import QAAnswer, QACitation
from codecompass.rag import RAGContext


def test_format_answer_prints_verified_citations() -> None:
    answer = QAAnswer(
        question="Where is escaping implemented?",
        answer="```\n[S3]\n[/S3]\nAnswer:\nThe escape function handles it. ```python\nreturn escape(value)\n```",
        citations=(QACitation("chunk-1", "src/package.py", "escape", "escape", 10, 20),),
        retrieval_method="hybrid",
        omitted_context_count=0,
        llm_model="local-model",
        llm_provider="ollama",
    )

    output = format_answer(answer)

    assert output.startswith("================================\nCodeQAPass Grounded QA Demo\n================================")
    assert "Question:\nWhere is escaping implemented?" in output
    assert "Answer:\nThe escape function handles it. return escape(value)" in output
    assert "```" not in output
    assert "[S3]" not in output
    assert "[/S3]" not in output
    assert output.count("Answer:") == 1
    assert "Model:\nlocal-model\n\nProvider:\nollama" in output
    assert "1. Symbol: escape\n   File: src/package.py\n   Lines: 10-20" in output


def test_format_answer_limits_display_without_changing_answer() -> None:
    citations = tuple(
        QACitation(f"chunk-{number}", f"src/file_{number}.py", f"symbol_{number}", None, number, number + 1)
        for number in range(1, 6)
    )
    answer = QAAnswer(
        question="Question",
        answer="Original answer",
        citations=citations,
        retrieval_method="hybrid",
        omitted_context_count=0,
        llm_model="model",
        llm_provider="provider",
    )

    output = format_answer(answer)

    assert "symbol_1" in output
    assert "symbol_3" in output
    assert "symbol_4" not in output
    assert "Additional evidence blocks not shown: 2" in output
    assert answer.answer == "Original answer"
    assert answer.citations == citations


def test_demo_prompt_adds_presentation_instructions() -> None:
    system_prompt, prompt = _DemoPromptBuilder().build("Which function escapes HTML?", RAGContext((), 0, 0))

    assert DEMO_RESPONSE_INSTRUCTIONS not in system_prompt
    assert "Question:\nWhich function escapes HTML?" in prompt
    assert DEMO_RESPONSE_INSTRUCTIONS in prompt
    assert "If the question is Persian, answer in Persian." in prompt


def test_format_answer_handles_fence_only_model_output() -> None:
    answer = QAAnswer(
        question="Question",
        answer="``` ``` ```",
        citations=(),
        retrieval_method="hybrid",
        omitted_context_count=0,
        llm_model="model",
        llm_provider="provider",
    )

    assert f"Answer:\n{NO_DISPLAYABLE_ANSWER}" in format_answer(answer)


def test_format_answer_displays_at_most_two_sentences_without_changing_answer() -> None:
    raw_answer = "First sentence. Second sentence! Third sentence?"
    answer = QAAnswer(
        question="Question",
        answer=raw_answer,
        citations=(),
        retrieval_method="hybrid",
        omitted_context_count=0,
        llm_model="model",
        llm_provider="provider",
    )

    output = format_answer(answer)

    assert "First sentence. Second sentence!" in output
    assert "Third sentence?" not in output
    assert answer.answer == raw_answer


def test_display_prioritizes_symbol_named_by_question_without_changing_citations() -> None:
    citations = (
        QACitation("markup", "src/package.py", "Markup", "Markup", 20, 40),
        QACitation("escape", "src/package.py", "escape", "escape", 1, 10),
        QACitation("unescape", "src/package.py", "unescape", "Markup.unescape", 30, 35),
    )
    answer = QAAnswer(
        question="Which function escapes HTML text?",
        answer="Use escape.",
        citations=citations,
        retrieval_method="hybrid",
        omitted_context_count=0,
        llm_model="model",
        llm_provider="provider",
    )

    output = format_answer(answer)

    assert output.index("1. Symbol: escape") < output.index("2. Symbol: Markup")
    assert answer.citations == citations
