import time
import os
import queue
import numpy as np
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from src.audio import iniciar_captura, parar_captura, SileroVAD

from src.stt import transcrever_audio

from src.llm.client import LLMClient

from src.tts.engine import TTSEngine
from src.tts.player import AudioPlayer
from src.tts.buffer import StreamBuffer

from faster_whisper import WhisperModel

# Carrega as variáveis do arquivo .env para a memória
load_dotenv()

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "jarvis-q5km")  # Nome do modelo LLM configurado no .env
# Inicializa a LLM (escolha o modelo carregado no seu Ollama local
llm = LLMClient(model_name=LLM_MODEL_NAME)

# Inicializa o TTS (Piper) com o modelo e configuração especificados no .env
tts = TTSEngine(
    model_path=os.getenv("TTS_MODEL_PATH", "models/tts/pt_BR-cadu-medium.onnx"),
    config_path=os.getenv("TTS_CONFIG_PATH", "models/tts/pt_BR-cadu-medium.onnx.json")
)

# Instancie o executor no setup inicial do main.py
tts_executor = ThreadPoolExecutor(max_workers=2)

# Inicializa o player de áudio para reproduzir o áudio sintetizado
player = AudioPlayer()

# Captura o tamanho do modelo configurado no .env (se não achar, usa 'base' como padrão)
model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
device = os.getenv("WHISPER_MODEL_DEVICE", "cpu")
compute_type = os.getenv("WHISPER_MODEL_COMPUTE_TYPE", "int8")

print(f"[Jarvis] Inicializando o modelo de transcrição: {model_size}")

def main():
    # Inicializa o Whisper
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    # Inicializa o VAD
    vad = SileroVAD()
    # Cria uma fila para guardar os chunks
    fila_de_audio = queue.Queue()
    # Buffer de áudio
    buffer_fala = []
    gravando_frase = False
    iniciar_captura(fila_de_audio) #Liga o microfone em segundo plano

    # Callback disparada sempre que o StreamBuffer fecha uma frase
    def process_sentence(sentence: str):
        # Sintetiza o texto em RAM
        def _synthesize_task(text):
            audio_data, sample_rate = tts.synthesize(sentence)
            print(f"\n[TTS Debug] Frase: '{sentence}' | Amostras: {len(audio_data)} | Sample Rate: {sample_rate}")
            # Envia para a fila do player
            player.add_audio(audio_data, sample_rate)

        tts_executor.submit(_synthesize_task, sentence)


    player.start()  # Inicia a thread de reprodução de áudio

    # Inicializa o buffer por frase
    stream_buffer = StreamBuffer(on_sentence_complete=process_sentence)

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
                            print("🤖 JARVIS: ", end="", flush=True)

                            # Consome o stream da LLM
                            for chunk in llm.generate_stream(texto_completo):
                                print(chunk, end="", flush=True)
                                stream_buffer.process_token(chunk)

                            print("\n")
                            # Finaliza a resposta: processa sobras do buffer e aguarda o áudio terminar
                            stream_buffer.flush()
                            player.audio_queue.join()  # Aguarda o áudio terminar de tocar antes de continuar
                        else:
                            print("📝 Transcrição: (Áudio incompreensível ou silêncio)")

                    
                    # Limpa o buffer para a próxima frase e reseta o VAD
                    buffer_fala.clear()
                    vad.resetar()


    except KeyboardInterrupt:
        print("\n[INFO] Encerrando aplicação...")
    finally:
        player.stop_and_wait() # Encerra o player ao sair da aplicação
        parar_captura() #Desliga microfone

    return


if __name__ == "__main__":
    main()