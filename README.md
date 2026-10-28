# JARVIS - Local Voice Assistant

JARVIS(*nome provisório) é um assistente virtual de voz modular, ultrarrápido e 100% local, projetado para execução em hardware de consumo com foco em baixa latência (TTFT reduzido) e alto desempenho.

---

## Arquitetura e Pipeline

A pipeline de processamento é assíncrona/streamed e opera no fluxo:

```text
[ Microfone ] ──> [ Silero VAD ] ──> [ Faster-Whisper ] ──> [ Ollama (LLM) ] ──> [ Resposta ]
  (Entrada)      (Filtro Voz)        (STT - CPU)        (Streaming)       (Terminal/TTS)
```

1. **VAD (Voice Activity Detection):** Captura de áudio direto em RAM filtrando silêncio via `Silero VAD`.
2. **STT (Speech-to-Text):** Transcrição de alta precisão em português utilizando `faster-whisper` em CPU (`float32`).
3. **LLM Engine:** Processamento e geração de linguagem natural em tempo real via API local do `Ollama` (`Qwen2.5`).

---

## Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **VAD:** Silero VAD / PyTorch
* **STT:** `faster-whisper` (`CTranslate2`)
* **LLM Engine:** `ollama-python`
* **Áudio:** `sounddevice`, `numpy`

---

## Pré-requisitos

1. **Python 3.10** ou superior instalado.
2. **Ollama** instalado e rodando localmente:
```bash
ollama pull qwen2.5
ollama serve
```

3. Microfone funcional e configurado como dispositivo padrão no SO.

---

## Como Rodar o Projeto

1. **Clone o repositório:**
```bash
git clone https://github.com/ArthurFaturini/local-voice-assistant.git
cd local-voice-assistant
```

2. **Crie e ative um ambiente virtual:**
```bash
python -m venv venv

# No Windows:
.\venv\Scripts\activate

# No Linux/Mac:
source venv/bin/activate
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Configuração de Variáveis de Ambiente**
```bash
# No Linux/Mac:
cp .env.example .env

# No Windows (PowerShell):
copy .env.example .env
```

5. **Inicie o assistente:**
```bash
python main.py
```

---

## Estrutura do Projeto

```text
jarvis/
├── src/
│   ├── audio/          # Captura de microfone e integração com VAD
│   ├── stt/            # Transcrição de áudio com Faster-Whisper
│   └── llm/            # Cliente Ollama, histórico e prompts de sistema
├── main.py             # Script principal e orquestrador da pipeline
├── requirements.txt    # Dependências do projeto
├── .env.example        # Modelo de variáveis de ambiente
├── .gitignore
└── README.md           # Documentação
```

---

## Próximos Passos (Roadmap)

* [ ] **Síntese de Voz (TTS):** Integrar engine de conversão de texto em fala (*Text-to-Speech*) para reprodução ágil do áudio gerado via streaming.
* [ ] **Tool Calling / Automações:** Adicionar suporte a execução de scripts Python locais para automação de tarefas do ambiente de trabalho.

---

## Autor

Desenvolvido com ☕ por **Arthur Faturini**.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arthur-faturini)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ArthurFaturini)