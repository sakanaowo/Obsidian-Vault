# Lesson Template — AI Tutor

Guide for structuring each lesson based on student configuration.

## Lesson Structure

Every lesson MUST follow this exact structure, adapted to the student's configured style.

### 1. Internal Thoughts (not shown to student unless Socratic mode)

Before presenting the lesson, mentally determine:

- What does the student already know? (based on depth + completed topics)
- What analogies/style fit their learning_style?
- What reasoning path fits their reasoning_framework?
- Are there prerequisite gaps to address first?

### 2. Lesson Header

```markdown
## [Topic Name]

**Buổi:** [session label if applicable]
**Mục tiêu:** Sau bài này, bạn sẽ hiểu [specific learning outcome].
**Prerequisites:** [[Concept A]], [[Concept B]]
```

### 3. Content — 3-Tier Structure (MANDATORY)

#### Tier 1 — ELI5

```markdown
> [!NOTE] ELI5
> [2-5 sentences. Everyday analogy. No jargon. Vietnamese.]
```

Rules for ELI5:

- Use analogies from daily life (cooking, driving, building, etc.)
- Must be understandable by a 5-year-old
- Must be technically accurate (simplified, not wrong)

#### Tier 2 — Technical Definition

Immediately after ELI5. A short paragraph covering:

- **Đây là gì?** — 1-2 sentence precise definition
- **Nó làm gì?** — Input → Output (with shapes/types if applicable)
- **Tại sao cần nó?** — What problem it solves, what it replaces

#### Tier 3 — Deep Mechanism

Subsections covering:

- Mathematical formulation (LaTeX)
- Code implementation (Python/PyTorch, comments in Vietnamese)
- Data flow / architectural diagram (describe or reference image)
- Comparison with related concepts
- Common mistakes / gotchas (`> [!WARNING]` callout)

### 4. Example Problem

```markdown
### Ví dụ minh họa

**Bài toán:** [Concrete problem statement]

**Lời giải:** [Step-by-step solution matching student's reasoning_framework]
```

Adapt based on `communication_style`:

- **Socratic:** Present the problem, ask the student to solve first, then discuss
- **Textbook:** Full worked solution with formal notation
- **Layman:** Conversational walkthrough
- **Story Telling:** Frame as a narrative scenario

### 5. Summary

```markdown
### Tóm tắt

- **Key takeaway 1:** [...]
- **Key takeaway 2:** [...]
- **Liên hệ:** Concept này kết nối với [[Related Concept]] vì [reason].
```

### 6. Post-Lesson Actions

```markdown
---

**Tiếp theo:** [next topic in curriculum]
**Gợi ý:**

1. [Follow-up question the student might want to explore]
2. [Another direction to deepen understanding]

Dùng `/tutor-continue` để sang bài tiếp, hoặc `/tutor-test` để kiểm tra.
```

## Style Adaptation Guide

### By Learning Style

| Style      | Prioritize                                                | Avoid                               |
| ---------- | --------------------------------------------------------- | ----------------------------------- |
| Visual     | Diagrams, plots, architecture drawings, data flow         | Walls of text without visual breaks |
| Verbal     | Clear prose, verbal explanations, reading-style content   | Over-reliance on diagrams           |
| Active     | Hands-on code examples, "try this" exercises              | Passive reading                     |
| Intuitive  | Big picture first, then details; patterns and connections | Starting with isolated details      |
| Reflective | Time to think, journaling prompts, comparison tables      | Rushing through content             |

### By Communication Style

| Style         | Approach                                                     |
| ------------- | ------------------------------------------------------------ |
| Formal        | Academic tone, precise terminology, structured argumentation |
| Textbook      | Textbook-style: definition → theorem → proof → example       |
| Layman        | Casual, approachable, lots of everyday language              |
| Socratic      | Ask questions before revealing answers; guide discovery      |
| Story Telling | Frame concepts as narratives with characters and conflict    |

### By Depth

| Depth         | Math Level                         | Code Level             | Detail Level           |
| ------------- | ---------------------------------- | ---------------------- | ---------------------- |
| Elementary    | Arithmetic only                    | No code                | High-level intuition   |
| High School   | Basic algebra                      | Simple Python          | Core ideas + examples  |
| Undergraduate | Calculus, linear algebra           | NumPy/basic PyTorch    | Full derivations       |
| Graduate      | Multivariate calc, probability     | Full PyTorch           | Implementation details |
| Master's      | Advanced optimization              | Research-grade code    | Paper-level depth      |
| Ph.D          | Measure theory, information theory | Custom implementations | Cutting-edge nuance    |

### By Reasoning Framework

| Framework  | Structure                                              |
| ---------- | ------------------------------------------------------ |
| Deductive  | General principle → specific case → conclusion         |
| Inductive  | Specific examples → pattern recognition → general rule |
| Abductive  | Observation → best explanation → verification          |
| Analogical | Known concept → analogy mapping → new concept          |
| Causal     | Cause → mechanism → effect → implications              |
