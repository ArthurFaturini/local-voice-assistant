import time
from faster_whisper import WhisperModel
import numpy as np

prompt_contexto = (
        "Jarvis, Python, VS Code, Git, GitHub, Pull Request, Commit, Merge, Branch,"
        "VAD, Silero, Whisper, STT, LLM, NumPy, RAM, Asyncio, API, Docker, Termux, "
        "Tailscale, JSON, Script, RU, gás, Chrome, tokens, IA local, output"
    )

def transcrever_audio(modelo, buffer_audio):
    # Garante que o áudio seja Unidimensional (1D / Mono)
    # Se tiver mais de uma dimensão (ex: estéreo), pega apenas o primeiro canal ou achata
    if buffer_audio.ndim > 1:
        # Se for formato (amostras, canais), pega o primeiro canal
        if buffer_audio.shape[1] < buffer_audio.shape[0]:
            buffer_audio = buffer_audio[:, 0]
        else:
            buffer_audio = buffer_audio.flatten()
            
    # Garante que o tipo de dado seja float32 e esteja normalizado
    if buffer_audio.dtype != np.float32:
        if buffer_audio.dtype == np.int16:
            buffer_audio = buffer_audio.astype(np.float32) / 32768.0
        else:
            buffer_audio = buffer_audio.astype(np.float32)
    
    # Normalização básica caso os valores estejam fora do padrão -1.0 a 1.0
    buffer_audio = buffer_audio - np.mean(buffer_audio)  # Remove DC offset
    
    max_val = np.max(np.abs(buffer_audio))
    if max_val > 1e-6:
        buffer_audio /= max_val

    # Limita o áudio por segurança (Impede que lixo acumulado quebre o código)
    # Se o áudio for menor que 0.5 segundos, nem tenta transcrever
    if len(buffer_audio) < 8000:  
        return [], None

    # Registra o momento exato antes de iniciar a IA
    tempo_inicial = time.time()

    # Executa a transcrição direto da memória RAM
    segments, info = modelo.transcribe(
        buffer_audio, 
        beam_size=5, 
        language="pt", 
        initial_prompt=prompt_contexto,
        temperature=0.0,
        best_of=5,
        compression_ratio_threshold=2.4)

    # Ele precisa consumir o gerador para realizar o processamento completo antes de medir o tempo
    segments = list(segments) 
    
    # Calcula quantos segundos se passaram
    tempo_gasto = time.time() - tempo_inicial
    
    # Exibe no terminal o tempo de processamento e a duração do áudio falado
    duracao_audio = len(buffer_audio) / 16000  # Calcula a duração em segundos (assumindo 16kHz)
    print(f"⏱️ [Whisper] Áudio de {duracao_audio:.2f}s processado em {tempo_gasto:.2f}s!")
    
    return segments, info
