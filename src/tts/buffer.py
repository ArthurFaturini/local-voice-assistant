import re
from typing import Callable

class StreamBuffer:
    def __init__(self, on_sentence_complete: Callable[[str], None]):
        """
        :param on_sentence_complete: Callback chamado sempre que uma frase completa é montada.
                                     Recebe a string da frase limpa.
        """
        self.on_sentence_complete = on_sentence_complete
        self.buffer = ""
        # Expressão regular para identificar delimitadores de frase
        self.sentence_end_pattern = re.compile(r'([.!?;\n]+)')

    def process_token(self, token: str):
        """
        Recebe cada token individual vindo do stream da LLM e acumula no buffer.
        """
        self.buffer += token

        # Verifica se existe pelo menos um pontuador no buffer atual
        match = self.sentence_end_pattern.search(self.buffer)
        if match:
            # Encontra o índice onde a frase termina (incluindo a pontuação)
            end_idx = match.end()
            
            # Extrai a frase completa e mantém o resto no buffer
            sentence = self.buffer[:end_idx].strip()
            self.buffer = self.buffer[end_idx:]

            # Se a frase isolada for muito curta (ex: "Olá!"), só envia se o buffer total
            # tiver acumulado contexto suficiente, ou aguarda acumular mais tokens.
            if len(sentence) >= 10 or any(p in sentence for p in ['!', '?', '.']):
                self.buffer = self.buffer[end_idx:]
                cleaned_sentence = self._clean_text(sentence)
                if cleaned_sentence:
                    self.on_sentence_complete(cleaned_sentence)

    def flush(self):
        """
        Força o envio de qualquer texto restante no buffer ao final do stream da LLM.
        """
        remaining = self.buffer.strip()
        if remaining:
            cleaned_sentence = self._clean_text(remaining)
            if cleaned_sentence:
                self.on_sentence_complete(cleaned_sentence)
        self.buffer = ""

    def _clean_text(self, text: str) -> str:
        """
        Sanitiza o texto removendo marcações de markdown (ex: **, *, #)
        para evitar que o TTS tente ler símbolos de formatação.
        """
        # Remove negritos/italicos (* ou _)
        text = re.sub(r'[\*_]+', '', text)
        # Remove marcadores de código ou pontuações excessivas se necessário
        text = re.sub(r'`+', '', text)
        return text.strip()