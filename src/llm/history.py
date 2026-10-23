from typing import List, Dict

class MessageHistory:
    def __init__(self, system_prompt: str, max_messages: int = 10):
        self.max_messages = max_messages
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

    def _trim_history(self):
        # Mantém o system prompt intacto na posição 0 e preserva apenas as últimas N mensagens
        if len(self.messages) > (self.max_messages + 1):
            self.messages = [self.messages[0]] + self.messages[-self.max_messages:]