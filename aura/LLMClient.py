from abc import ABC, abstractmethod


TIMEOUT = 3600


class LLMClient(ABC):
    def __init__(self, host: str, port: int, system_prompt: str, label: str):
        self.host = host
        self.port = port
        self.system_prompt = system_prompt
        self.label = label

    @property
    @abstractmethod
    def url(self) -> str:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def complete(self, prompt: str, add_to_history: bool = True) -> str:
        pass
