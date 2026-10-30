import queue
import threading
import sounddevice as sd
import numpy as np

class AudioPlayer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self._stop_event = threading.Event()
        self._worker_thread = None

    def start(self):
        """Inicia a thread trabalhadora responsável por consumir a fila de áudio."""
        self._stop_event.clear()
        self.is_playing = True
        self._worker_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._worker_thread.start()

    def add_audio(self, audio_data: np.ndarray, sample_rate: int):
        """Adiciona um bloco de áudio (frase sintetizada) à fila de reprodução."""
        if audio_data.size > 0:
            self.audio_queue.put((audio_data, sample_rate))

    def _play_loop(self):
        """Loop executado na thread secundária que reproduz os áudios da fila."""
        while not self._stop_event.is_set():
            try:
                # Aguarda até existir um bloco de áudio na fila (timeout evita bloqueio perpétuo)
                audio_data, sample_rate = self.audio_queue.get(timeout=0.1)
                
                # Reproduz o bloco de áudio de forma síncrona dentro da thread dedicada
                sd.play(audio_data, samplerate=sample_rate)
                sd.wait()  # Aguarda a frase atual terminar antes de pegar na próxima
                
                self.audio_queue.task_done()
            except queue.Empty:
                continue

    def stop_and_wait(self):
        """Aguardar que todos os áudios da fila terminem de tocar antes de libertar o dispositivo."""
        self.audio_queue.join()  # Bloqueia até a fila esvaziar completamente
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join()
        sd.stop()
        self.is_playing = False