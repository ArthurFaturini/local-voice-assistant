import os
import io
import wave
import numpy as np
from piper import PiperVoice

class TTSEngine:
    def __init__(self, model_path: str, config_path: str):
        """
        Inicializa o wrapper do Piper carregando o modelo ONNX em memória.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo TTS não encontrado em: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuração TTS não encontrada em: {config_path}")

        # Carrega a voz do Piper
        self.voice = PiperVoice.load(model_path, config_path=config_path)
        self.sample_rate = self.voice.config.sample_rate

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not text.strip():
            return np.array([], dtype=np.float32), self.sample_rate

        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            self.voice.synthesize_wav(text, wav_file)

        wav_buffer.seek(0)

        with wave.open(wav_buffer, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        # Sem trimming! Devolve o áudio completo com as caudas naturais de som.
        return audio_data, self.sample_rate