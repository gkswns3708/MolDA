---
paths:
  - "src/data/**"
  - "src/model/**"
  - "src/training/**"
  - "src/generation/**"
---

# Pipeline 변경 시 테스트 필수

파이프라인 로직(data, model, training, generation)을 수정할 때:
1. 해당 변경의 정합성을 검증하는 테스트를 `tests/`에 작성 또는 업데이트한다
2. 새 함수/클래스 추가 시 최소 1개의 unit test를 포함한다
3. config 변경이 코드에 영향을 주는 경우, 해당 config 값이 올바르게 반영되는지 테스트한다
4. 테스트 없이 파이프라인 변경을 완료하지 않는다
