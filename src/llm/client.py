import time
from typing import Generator
import ollama
from .history import MessageHistory
from .system_prompt import SYSTEM_PROMPT

class LLMClient:
    def __init__(self, model_name: str = "gemma3:1b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.client = ollama.Client(host=host)
        self.history = MessageHistory(system_prompt=SYSTEM_PROMPT)

    def generate_stream(self, user_text: str) -> Generator[str, None, str]:
        """
        Envia o texto do usuário para o Ollama em modo Streaming.
        Cede (yield) os fragmentos de texto em tempo real conforme chegam.
        """
        # Adiciona a fala do usuário ao histórico
        self.history.add_user_message(user_text)

        inicio_requisicao = time.time()
        tempo_primeiro_token = None
        eval_count = 0
        eval_duration = 0

        # Chama a API do Ollama com stream=True
        response_stream = self.client.chat(
            model=self.model_name,
            messages=self.history.get_messages(),
            stream=True,
            think=False,  # Desativa o "pensamento" da IA para respostas mais rápidas
            keep_alive="-1m"  # Mantém o modelo preso na VRAM para resposta instantânea
        )

        full_response = ""

        # Itera sobre os pedaços (chunks) recebidos em tempo real
        for chunk in response_stream:
            # Marca o Time To First Token (TTFT) no primeiro pedaço recebido
            if tempo_primeiro_token is None:
                tempo_primeiro_token = time.time() - inicio_requisicao

            content = chunk['message']['content']
            if content:
                full_response += content
                yield content  # Envia o trecho de texto imediatamente

            # Captura métricas do Ollama ao final do stream
            if chunk.get('done'):
                eval_count = chunk.get('eval_count', 0)          # total de tokens gerados
                eval_duration = chunk.get('eval_duration', 1)    # tempo em nanosegundos

        tempo_total = time.time() - inicio_requisicao
        # Salva a resposta completa gerada no histórico para manter contexto
        self.history.add_assistant_message(full_response)

        # Calcula tokens por segundo (eval_duration vem em nanosegundos)
        tokens_por_segundo = (eval_count / (eval_duration / 1e9)) if eval_duration > 0 else 0

        print(f"\n[LLM] Tempo total: {tempo_total:.2f}s, TTFT: {tempo_primeiro_token:.2f}s, Tokens: {eval_count}, TPS: {tokens_por_segundo:.2f}")
        
        return full_response