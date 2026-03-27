"""Tests for special token definitions and LLaDA token ID consistency.

검증 대상:
- src/model/added_tokens.py — 태그 토큰 정의
- src/training/loss.py — MASK_TOKEN_ID
- src/data/collator.py — TrainCollator, EvalCollator (토크나이저 기반 EOS/PAD)
- 실제 토크나이저 — 토큰 등록 및 인덱스 경계
"""

from src.model import added_tokens
from src.training.loss import MASK_TOKEN_ID
from src.data.collator import TrainCollator, EvalCollator


def _all_tokens():
    """Collect all tokens from all lists."""
    return (
        added_tokens.BOOL + added_tokens.FLOAT + added_tokens.DESCRIPTION
        + added_tokens.SELFIES + added_tokens.MOL_2D + added_tokens.MOL_3D
        + added_tokens.MOL_EMBEDDING + added_tokens.NUMBER
        + added_tokens.INSTRUCTION + added_tokens.REACTION_DIRECTION
        + added_tokens.IUPAC + added_tokens.MOLFORMULA
    )


# ── LLaDA 기본 토큰 상수 ──
ORIGINAL_VOCAB_SIZE = 126349


class TestAddedTokens:
    """added_tokens.py 정의 자체의 무결성."""

    def test_all_tokens_are_strings(self):
        for token in _all_tokens():
            assert isinstance(token, str), f"Token {token!r} is not a string"

    def test_no_duplicate_tokens(self):
        all_tok = _all_tokens()
        assert len(all_tok) == len(set(all_tok)), (
            f"Duplicate tokens found: {[t for t in all_tok if all_tok.count(t) > 1]}"
        )

    def test_paired_tags_have_open_and_close(self):
        paired = [
            added_tokens.BOOL, added_tokens.FLOAT, added_tokens.DESCRIPTION,
            added_tokens.SELFIES, added_tokens.MOL_2D, added_tokens.MOL_3D,
            added_tokens.INSTRUCTION, added_tokens.IUPAC, added_tokens.MOLFORMULA,
        ]
        for pair in paired:
            assert len(pair) == 2, f"Expected pair, got {pair}"
            assert pair[0].startswith("<"), f"Open tag should start with '<': {pair[0]}"
            assert pair[1].startswith("</"), f"Close tag should start with '</': {pair[1]}"

    def test_number_tokens_complete(self):
        nums = added_tokens.NUMBER
        assert len(nums) == 13, f"Expected 13 NUMBER tokens (0-9 + +, -, .), got {len(nums)}"
        for i in range(10):
            assert f"<|{i}|>" in nums
        assert "<|+|>" in nums
        assert "<|-|>" in nums
        assert "<|.|>" in nums

    def test_mol_embedding_token(self):
        assert "<mol>" in added_tokens.MOL_EMBEDDING

    def test_reaction_direction_token(self):
        assert "|>>|" in added_tokens.REACTION_DIRECTION

    def test_added_token_count(self):
        """태그 토큰 총 33개: 태그 쌍 9×2=18 + NUMBER 13 + <mol> 1 + |>>| 1 = 33."""
        assert len(_all_tokens()) == 33


class TestLLaDATokenIDs:
    """코드에 하드코딩된 특수 토큰 ID 상수들의 값과 관계 검증."""

    def test_mask_token_id(self):
        """<|mdm_mask|> = 126336 (LLaDA 공식 값)."""
        assert MASK_TOKEN_ID == 126336

    def test_mask_within_original_vocab(self):
        """MASK 토큰은 기본 vocab 범위(< 126349) 안에 있다."""
        assert MASK_TOKEN_ID < ORIGINAL_VOCAB_SIZE


class TestTokenizerIntegration:
    """실제 토크나이저에서 토큰 등록 및 인덱스 경계 검증."""

    def test_tokenizer_eos_token(self, real_tokenizer):
        """토크나이저의 eos = <|endoftext|> (id=126081)."""
        assert real_tokenizer.eos_token == "<|endoftext|>"
        assert real_tokenizer.eos_token_id == 126081

    def test_tokenizer_pad_token(self, real_tokenizer):
        """토크나이저 기본 pad = eos와 동일 (<|endoftext|>)."""
        # LLaDA 토크나이저는 기본적으로 pad=eos
        assert real_tokenizer.pad_token_id == real_tokenizer.eos_token_id

    def test_collator_eos_matches_tokenizer(self, real_tokenizer, cfg):
        """Regression: collator EOS는 토크나이저 eos_token_id와 일치해야 한다."""
        collator = TrainCollator(real_tokenizer, max_length=cfg.data.max_length)
        assert collator.eos_token_id == real_tokenizer.eos_token_id

    def test_collator_pad_derives_from_tokenizer(self, real_tokenizer, cfg):
        """Regression: collator PAD는 토크나이저 pad_token_id에서 파생되어야 한다."""
        collator = EvalCollator(real_tokenizer, max_length=cfg.data.max_length)
        expected = real_tokenizer.pad_token_id if real_tokenizer.pad_token_id is not None else real_tokenizer.eos_token_id
        assert collator.pad_token_id == expected

    def test_mask_token_decodable(self, real_tokenizer):
        """MASK 토큰 ID(126336)가 디코딩 가능."""
        decoded = real_tokenizer.decode([MASK_TOKEN_ID])
        assert len(decoded) > 0

    def test_mask_token_is_mdm_mask(self, real_tokenizer):
        """MASK 토큰이 <|mdm_mask|>로 디코딩된다."""
        decoded = real_tokenizer.decode([MASK_TOKEN_ID]).strip()
        assert "mdm_mask" in decoded, f"Expected '<|mdm_mask|>', got {decoded!r}"

    def test_added_tokens_start_at_original_vocab(self, real_tokenizer):
        """추가된 토큰의 ID는 original_vocab_size(126349)부터 시작."""
        for token_str in _all_tokens():
            token_id = real_tokenizer.convert_tokens_to_ids(token_str)
            assert token_id >= ORIGINAL_VOCAB_SIZE, (
                f"Added token {token_str!r} has id {token_id} "
                f"< original_vocab_size {ORIGINAL_VOCAB_SIZE}"
            )

    def test_added_tokens_contiguous(self, real_tokenizer):
        """추가된 태그 토큰들의 ID가 126349부터 연속 블록으로 할당."""
        ids = sorted(
            real_tokenizer.convert_tokens_to_ids(t) for t in _all_tokens()
        )
        expected_start = ORIGINAL_VOCAB_SIZE
        assert ids[0] == expected_start, (
            f"First added token id: {ids[0]}, expected {expected_start}"
        )
        assert ids[-1] == expected_start + len(ids) - 1, (
            f"Added token ids not contiguous: first={ids[0]}, last={ids[-1]}, count={len(ids)}"
        )

    def test_vocab_size_after_expansion(self, real_tokenizer):
        """확장 후 vocab > original_vocab_size."""
        assert len(real_tokenizer) > ORIGINAL_VOCAB_SIZE

    def test_mol_token_registered(self, real_tokenizer):
        """<mol> 토큰이 unknown이 아닌 고유 ID로 등록."""
        mol_id = real_tokenizer.convert_tokens_to_ids("<mol>")
        unk_id = real_tokenizer.convert_tokens_to_ids("<unk>")
        assert mol_id != unk_id, "<mol> resolved to <unk>"

    def test_original_vocab_boundary(self, real_tokenizer):
        """original_vocab_size 경계: idx 126348은 기존 토큰, 126349는 추가 토큰."""
        # 126348(마지막 기존 토큰)은 added_tokens에 없어야 한다
        added_ids = {
            real_tokenizer.convert_tokens_to_ids(t) for t in _all_tokens()
        }
        assert 126348 not in added_ids
        # 126349(첫 추가 토큰)는 added_tokens 중 하나여야 한다
        assert ORIGINAL_VOCAB_SIZE in added_ids
