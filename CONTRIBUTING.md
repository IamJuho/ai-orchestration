# Contributing Guide

이 저장소는 **review-friendly writing**을 중요하게 생각합니다.

목표는 단순히 change를 기록하는 것이 아니라, reviewer가 PR과 commit message만 읽어도
"무엇이 바뀌었고, 왜 그렇게 설계했는지"를 빠르게 파악할 수 있도록 만드는 것입니다.

## Writing Principles

### Korean-first, English-supported
- 기본 문체는 **자연스러운 한국어**를 사용합니다.
- 다만 technical term은 English를 섞어 쓰는 것이 더 명확하면 그대로 사용합니다.
- **영어-only writing은 피합니다.**

좋은 예:
- `runner timeout semantics를 terminal state 기준으로 정리한다`
- `typed contract 기반으로 worker handoff를 명확히 한다`

피해야 할 예:
- `Implemented improvements to the orchestration runtime.`

### Reviewer-first structure
repo-facing text는 reviewer가 빠르게 이해할 수 있어야 합니다.

특히 다음을 먼저 설명합니다:
- What changed
- Why this change exists
- End-to-end flow
- Boundaries / Non-goals
- Testing evidence

## Commit Message Rules

Commit message는 **읽는 사람이 바로 이해할 수 있는 문장형 subject**를 사용합니다.

권장 원칙:
- Korean-first with selective English technical terms
- vague wording 금지
- one commit = one understandable unit
- implementation detail보다 reviewer intent 우선

좋은 예:
- `Runner timeout semantics를 worker call 기준으로 정리한다`
- `PR writing rule과 policy workflow를 repo-level로 고정한다`

피해야 할 예:
- `fix stuff`
- `update files`
- `refactor code`

## Pull Request Writing Rules

PR 본문은 reviewer가 change set의 흐름을 재구성하지 않아도 되도록 아래 구조를 따릅니다.

### Required sections
- `## 요약`
- `## 왜 이 변경이 필요한가`
- `## 전체 흐름 / End-to-end flow`
- `## 주요 변경점`
- `## 테스트 / Verification`
- `## 범위와 비목표 / Scope & Non-goals`
- `## reviewer가 특히 봐야 할 포인트`

### PR writing guidance
- file listing보다 **system flow**를 먼저 설명합니다.
- architecture와 runtime behavior를 분리해서 설명합니다.
- happy path와 failure semantics가 있으면 반드시 적습니다.
- 테스트는 command 기준으로 적고, 무엇이 offline / live인지 구분합니다.

## Repo-facing Docs Rules

README, PR body, release note, migration note 같은 repo-facing 문서는:
- 한국어 중심으로 작성하고
- 필요한 English technical term만 혼용하며
- reviewer 또는 maintainer가 재질문하지 않도록 맥락을 충분히 제공합니다.

## AI Attribution Policy

아래와 같은 **AI attribution marker**는 commit message, PR title/body, README, docs, template 같은 repo-facing writing에 넣지 않습니다:

- generated marker
- AI writing marker
- AI agent co-author trailer
- 이와 유사하게 AI 작성 사실을 직접 드러내는 표현

이 저장소의 repo-facing writing은 결과물 기준으로 읽혀야 하며,
작성 도구 자체가 문서/commit/PR의 일부가 되면 안 됩니다.

## Enforcement

이 규칙은 다음 방식으로 유지합니다:
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/workflows/policy.yml`
- project notepad decisions

현재 workflow enforcement 범위는 다음과 같습니다:
- tracked markdown / PR template의 금지 문구 검사
- pushed commit message 전체 텍스트의 금지 문구 검사
- pushed commit subject의 Korean-first 검사
- PR title/body의 금지 문구 + required section 검사

즉, guidance + template + CI policy를 함께 사용합니다.
