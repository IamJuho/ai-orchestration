## 요약
- 이 PR이 무엇을 바꾸는지 2~4 bullet로 설명해주세요.
- 한국어를 반드시 포함하고, 필요한 English technical term만 섞어주세요.

## 왜 이 변경이 필요한가
- 배경 / 문제 / 의도를 설명해주세요.
- reviewer가 "왜 지금 이 변경이 필요한가"를 바로 이해할 수 있어야 합니다.

## 전체 흐름 / End-to-end flow
- 가능하면 아래처럼 flow를 한 번에 설명해주세요.

예시:
`CLI -> Settings/Provider -> Runner -> Orchestrator -> Worker Registry -> FinalResponse`

## 주요 변경점
### Runtime / Architecture
-

### Data / Contracts
-

### Testing / Ops
-

## 테스트 / Verification
실제로 실행한 command를 적어주세요.

```bash
# example
python -m pip install -e ".[dev]"
uv run --extra dev ruff format --check .
uv run --extra dev ruff check .
uv run --extra dev mypy src tests
uv run --extra dev pytest -m "not live"
```

추가 검증이 있으면 구분해서 적어주세요.
- offline:
- live / opt-in:
- manual QA:

## 범위와 비목표 / Scope & Non-goals
이번 PR에서 의도적으로 하지 않은 것을 적어주세요.

-

## reviewer가 특히 봐야 할 포인트
- 가장 risky한 runtime behavior
- regression 가능성이 있는 부분
- architecture seam / extension point

## 작성 체크리스트
- [ ] reviewer가 전체 흐름을 body만 읽고 이해할 수 있다
- [ ] 한국어를 포함했고, English-only writing을 피했다
- [ ] testing command를 실제 실행 기준으로 적었다
- [ ] out-of-scope를 명시했다
- [ ] AI attribution 문구를 포함하지 않았다
