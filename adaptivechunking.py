class AdaptiveChunker:

    def __init__(
        self,
        min_tokens: int = 50,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens


    @staticmethod
    def _approx_tokens(text: str) -> int:
   
        return int(len(text.split()) * 1.3)

    def _split_sentences(self, text: str) -> List[str]:
        return re.split(r"(?<=[.!?])\s+", text.strip())

    def chunk(self, text: str) -> List[str]:

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        raw_chunks: List[str] = []

        for para in paragraphs:
            if self._approx_tokens(para) <= self.max_tokens:
                raw_chunks.append(para)
            else:
            
                sentences = self._split_sentences(para)
                current: List[str] = []
                current_tokens = 0
                for sent in sentences:
                    sent_tokens = self._approx_tokens(sent)
                    if current_tokens + sent_tokens > self.max_tokens and current:
                        raw_chunks.append(" ".join(current))
            
                        overlap: List[str] = []
                        ov_tokens = 0
                        for s in reversed(current):
                            st = self._approx_tokens(s)
                            if ov_tokens + st > self.overlap_tokens:
                                break
                            overlap.insert(0, s)
                            ov_tokens += st
                        current = overlap + [sent]
                        current_tokens = self._approx_tokens(" ".join(current))
                    else:
                        current.append(sent)
                        current_tokens += sent_tokens
                if current:
                    raw_chunks.append(" ".join(current))

        chunks = [c for c in raw_chunks if self._approx_tokens(c) >= self.min_tokens]
        return self._add_overlap(chunks)

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1].split()
            overlap_words = prev_words[-self.overlap_tokens:] if len(prev_words) > self.overlap_tokens else prev_words
            overlap_text = " ".join(overlap_words)
            result.append(overlap_text + " " + chunks[i])
        return result