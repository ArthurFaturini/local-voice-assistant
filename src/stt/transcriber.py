from faster_whisper import WhisperModel
import numpy as np

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
            max_val = np.max(np.abs(buffer_audio))
            if max_val > 1.0:
                buffer_audio /= max_val

    # Limita o áudio por segurança (Impede que lixo acumulado quebre o código)
    # Se o áudio for menor que 0.5 segundos, nem tenta transcrever
    if len(buffer_audio) < 8000:  
        return [], None

    # Executa a transcrição direto da memória RAM
    segments, info = modelo.transcribe(buffer_audio, beam_size=5, language="pt")
    
    return segments, info
