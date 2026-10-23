import sounddevice as sd

fs = 16000 #taxa de amostragem
block_size = 512 #tamanho da chunk de audio
canais = 1
sd.default.channels = 1, 13 #define input e output

_stream = None #stream ativa

def callback_entrada(indata, frames, time, status, fila_de_audio):
    """Executada em segundo plano a cada novo chunk de áudio capturado."""
    if status:
        print(status, flush=True)
    # Coloca uma cópia do chunk de áudio na fila para processamento
    fila_de_audio.put(indata.copy())


def iniciar_captura(fila_de_audio):
    """Liga a captura de áudio em segundo plano."""
    global _stream
    if _stream is None:
        _stream = sd.InputStream(
            samplerate=fs, 
            channels=canais, 
            blocksize=block_size,
            callback=lambda indata, frames, time, status: 
            callback_entrada(indata, frames, time, status, fila_de_audio)
        )
        _stream.start() #Ativa o microfone
        print("🎙️ Microfone ligado em segundo plano...")


def parar_captura():
    """Desliga a captura de áudio."""
    global _stream
    if _stream is not None:
        _stream.stop()
        _stream.close()
        _stream = None
        print("Dispositivos desligados com sucesso!")