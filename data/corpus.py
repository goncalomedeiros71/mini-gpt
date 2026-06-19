import re


# Marcadores que delimitam o texto real numa eBook do Project Gutenberg.
_START = re.compile(r"\*\*\* START OF .*? \*\*\*", re.IGNORECASE)
_END   = re.compile(r"\*\*\* END OF .*? \*\*\*", re.IGNORECASE)


def clean_gutenberg(text):
    """Remove o cabeçalho e a licença do Project Gutenberg, deixando só a obra."""
    start = _START.search(text)
    end = _END.search(text)
    start_idx = start.end() if start else 0
    end_idx = end.start() if end else len(text)
    return text[start_idx:end_idx].strip()


def load_text(path="input.txt"):
    """Lê o ficheiro e devolve apenas o texto da obra (sem boilerplate)."""
    with open(path, "r", encoding="utf-8") as f:
        return clean_gutenberg(f.read())
