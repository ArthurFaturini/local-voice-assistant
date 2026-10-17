import time
import queue
from src import iniciar_captura, parar_captura


def main():
    # Cria uma fila para guardar os chunks
    fila_de_audio = queue.Queue()
    iniciar_captura(fila_de_audio) #Liga o microfone em segundo plano
    try:
        while True:
            #Se a fila tiver áudio
            if not fila_de_audio.empty():
                chunk = fila_de_audio.get()
                #Processar chunck com Silero VAD

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Encerrando aplicação...")
    finally:
        parar_captura() #Desliga microfone

    return


if __name__ == "__main__":
    main()