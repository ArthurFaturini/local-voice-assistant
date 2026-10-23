import time
import os
import queue
import numpy as np
from dotenv import load_dotenv
from src.audio import iniciar_captura, parar_captura, SileroVAD
from src.stt import transcrever_audio
from src.llm.client import LLMClient
from faster_whisper import WhisperModel

# Carrega as variáveis do arquivo .env para a memória
load_dotenv()

# Inicializa a LLM (escolha o modelo carregado no seu Ollama local
llm = LLMClient(model_name="jarvis-q5km")

# Captura o tamanho do modelo configurado no .env (se não achar, usa 'base' como padrão)
model_size = os.getenv("WHISPER_MODEL_SIZE", "base")

print(f"[Jarvis] Inicializando o modelo de transcrição: {model_size}")

def main():
    # Inicializa o Whisper
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
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
            try:
                # Pega o áudio da fila sem travar o loop para sempre
                chunk = fila_de_audio.get_nowait()
            except queue.Empty:
                time.sleep(0.01)
                continue

            if gravando_frase:
                buffer_fala.append(chunk)
        
            evento = vad.processar_chunk(chunk)

            if evento:
                if "start" in evento:
                    print("\n🎙️ [VAD] Fala detectada! Gravando...")
                    gravando_frase = True
                    buffer_fala.clear()
                    buffer_fala.append(chunk) 
                elif "end" in evento and gravando_frase:
                    print("🛑 [VAD] Fim da fala detectado!")
                    gravando_frase = False

                    if len(buffer_fala) > 0:
                        # Junta todos os chunks acumulados na RAM em um único array NumPy
                        audio_completo = np.concatenate(buffer_fala, axis=0)
                        print(f"⚡ Frase capturada na RAM com {len(audio_completo)} amostras.")

                        # Transcreve o áudio capturado
                        print("⏳ Transcrevendo...")
                        segments, info = transcrever_audio(model, audio_completo)
                        texto_completo = " ".join([segment.text for segment in segments]).strip()

                        if texto_completo:
                            print(f"📝 Transcrição: \"{texto_completo}\"")
                        else:
                            print("📝 Transcrição: (Áudio incompreensível ou silêncio)")

                        print("🤖 JARVIS: ", end="", flush=True)
                        # Consome o stream da LLM
                        for chunk in llm.generate_stream(texto_completo):
                            print(chunk, end="", flush=True)

                        print("\n")
                    
                    # Limpa o buffer para a próxima frase e reseta o VAD
                    buffer_fala.clear()
                    vad.resetar()


    except KeyboardInterrupt:
        print("\n[INFO] Encerrando aplicação...")
    finally:
        parar_captura() #Desliga microfone

    return


if __name__ == "__main__":
    main()