"""End-to-end graph-QA orchestration with retrieval/answer separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .answering import AnswerResult, MultipleChoiceAnswerer
from .program import QuestionProgram, RuleBasedProgramPlanner
from .retrieval import CompositionalGraphRetriever, RetrievalTrace
from .schema import TemporalGraph


@dataclass
class GraphQAResult:
    question_id: str
    graph_id: str
    category: str | None
    program: QuestionProgram
    retrieval: RetrievalTrace
    answer: AnswerResult

    def to_dict(self, include_prompt: bool = False) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "graph_id": self.graph_id,
            "category": self.category,
            "program": self.program.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "answer": self.answer.to_dict(include_prompt=include_prompt),
        }


class GraphQAPipeline:
    def __init__(
        self,
        *,
        retriever: CompositionalGraphRetriever,
        answerer: MultipleChoiceAnswerer,
        planner: RuleBasedProgramPlanner | None = None,
    ) -> None:
        self.retriever = retriever
        self.answerer = answerer
        self.planner = planner or RuleBasedProgramPlanner()

    def answer_question(
        self,
        *,
        graph: TemporalGraph,
        question_id: str,
        question: str,
        choices: Sequence[str],
        inputs: dict[str, Any] | None = None,
        category: str | None = None,
    ) -> GraphQAResult:
        program = self.planner.plan(
            question=question,
            inputs=inputs or {},
            category=category,
        )
        retrieval = self.retriever.retrieve(graph, program)
        answer = self.answerer.answer(program=program, choices=choices, retrieval=retrieval)
        return GraphQAResult(
            question_id=question_id,
            graph_id=graph.graph_id,
            category=category,
            program=program,
            retrieval=retrieval,
            answer=answer,
        )
