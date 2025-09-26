from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
from typing import Iterable, List, Tuple, Dict, Optional

_BOS = "<s>"
_EOS = "</s>"
_UNK = "<unk>"
_TOK = re.compile(r"[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)  # token + punctuation

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOK.findall(text or "")]

def _wrap_sents(texts: Iterable[str]) -> List[List[str]]:
    sents: List[List[str]] = []
    for t in texts:
        toks = _tokenize(t)
        if toks:
            sents.append([_BOS] + toks + [_EOS])
    return sents

class BigramModel:
    """Simple bigram language model with Laplace smoothing; supports generate_text()."""

    def __init__(self, corpus: Optional[Iterable[str]] = None, alpha: float = 1.0, min_freq: int = 1):
        self.alpha = float(alpha)
        self.min_freq = int(min_freq)
        self.vocab: Dict[str, int] = {}
        self.vocab_size = 0
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self._fitted = False
        if corpus is not None:
            self.fit(corpus)

    # ---------- training ----------
    def fit(self, texts: Iterable[str]) -> "BigramModel":
        raw = Counter()
        for t in texts:
            raw.update(_tokenize(t))

        self.vocab = {_BOS: 10**9, _EOS: 10**9, _UNK: 10**9}
        for w, c in raw.items():
            if c >= self.min_freq:
                self.vocab[w] = c
        self.vocab_size = len(self.vocab)

        self.unigram.clear()
        self.bigram.clear()
        for sent in _wrap_sents(texts):
            sent = [w if w in self.vocab else _UNK for w in sent]
            for i, w in enumerate(sent):
                self.unigram[w] += 1
                if i > 0:
                    self.bigram[sent[i - 1]][w] += 1

        self._fitted = True
        return self

    # ---------- probabilities ----------
    def _prob(self, prev: str, nxt: str) -> float:
        prev = prev if prev in self.vocab else _UNK
        nxt = nxt if nxt in self.vocab else _UNK
        num = self.bigram[prev][nxt] + self.alpha
        den = self.unigram[prev] + self.alpha * self.vocab_size
        return num / den if den > 0 else 1.0 / max(self.vocab_size, 1)

    # ---------- inference ----------
    def predict_next(self, prev_token: str, top_k: int = 5) -> List[Tuple[str, float]]:
        self._ensure_fitted()
        prev = prev_token.lower()
        if prev not in self.vocab and prev not in {_BOS, _EOS}:
            prev = _UNK
        cand: List[Tuple[str, float]] = []
        for w in self.vocab:
            if w == _BOS:
                continue
            cand.append((w, self._prob(prev, w)))
        cand.sort(key=lambda x: x[1], reverse=True)
        return cand[:max(1, top_k)]

    def generate_text(self, start_word: str, length: int = 10) -> str:
        """Greedy top-1 next-token generation."""
        self._ensure_fitted()
        length = max(1, int(length))
        prev = start_word.lower().strip()
        out_tokens: List[str] = [prev]
        if prev not in self.vocab:
            prev = _UNK

        for _ in range(length - 1):
            next_word, _ = max(
                ((w, self._prob(prev, w)) for w in self.vocab if w != _BOS),
                key=lambda x: x[1]
            )
            if next_word == _EOS:
                break
            out_tokens.append(next_word)
            prev = next_word

        return self._detok(out_tokens)

    @staticmethod
    def _detok(tokens: List[str]) -> str:
        s = ""
        for i, w in enumerate(tokens):
            if i > 0 and re.match(r"[^\w\s]", w) and w not in ("'",):
                s += w
            else:
                if s and not s.endswith(" "):
                    s += " "
                s += w
        return s.strip()

    # ---------- optional: scoring ----------
    def sentence_logprob(self, text: str) -> float:
        self._ensure_fitted()
        toks = [_BOS] + [w if w in self.vocab else _UNK for w in _tokenize(text)] + [_EOS]
        return sum(math.log(self._prob(toks[i - 1], toks[i])) for i in range(1, len(toks)))

    def perplexity(self, texts: Iterable[str]) -> float:
        self._ensure_fitted()
        total_logp = 0.0
        total_tokens = 0
        for t in texts:
            toks = [_BOS] + [w if w in self.vocab else _UNK for w in _tokenize(t)] + [_EOS]
            total_tokens += len(toks) - 1
            for i in range(1, len(toks)):
                total_logp += math.log(self._prob(toks[i - 1], toks[i]))
        if total_tokens == 0:
            return float("inf")
        return math.exp(-total_logp / total_tokens)

    # ---------- internal ----------
    def _ensure_fitted(self):
        if not self._fitted:
            raise RuntimeError("BigramModel is not trained; pass a corpus or call fit().")
