import time
import queue
import numpy as np
from src import iniciar_captura, parar_captura, SileroVAD


def main():
    # Inicializa o VAD
    vad = SileroVAD()
    # Cria uma fila para guardar os chunks
    fila_de_audio = queue.Queue()
    # Buffer de áudio
    buffer_fala = []
    gravando_frase = False
    iniciar_captura(fila_de_audio) #Liga o microfone em segundo plano
    try:
        while True:
            chunk = fila_de_audio.get()
            evento = vad.processar_chunk(chunk)

            if evento:
                if "start" in evento:
                    print("\n🎙️ [VAD] Fala detectada! Gravando...")
                    gravando_frase = True
                    buffer_fala.clear()
                elif "end" in evento:
                    print("🛑 [VAD] Fim da fala detectado!")
                    gravando_frase = False

                    # Junta todos os chunks acumulados na RAM em um único array NumPy
                    audio_completo = np.concatenate(buffer_fala, axis=0)

                    print(f"⚡ Frase capturada na RAM com {len(audio_completo)} amostras.")

                    vad.resetar()

            if gravando_frase:
                buffer_fala.append(chunk)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Encerrando aplicação...")
    finally:
        parar_captura() #Desliga microfone

    return


if __name__ == "__main__":
    main()