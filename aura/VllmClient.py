from json import dumps
from requests import post
from logging import getLogger

from .LLMClient import LLMClient, TIMEOUT


logger = getLogger(__name__)


class VllmClient(LLMClient):
    def __init__(self, host: str, port: int, model: str, system_prompt: str, label: str):
        super().__init__(system_prompt, label)

        self.host = host
        self.port = port

        self.model = model
        self.history = []

    @property
    def url(self):
        return f'http://{self.host}:{self.port}/v1/chat/completions'

    def reset(self):
        self.history = []

    def complete(self, prompt: str, add_to_history: bool = True):
        history = self.history

        if len(history) < 1:
            system_message = {
                'role': 'system',
                'content': self.system_prompt
            }
            init_message = {
                'role': 'user',
                'content': prompt
            }

            response = post(
                self.url,
                json = {
                    'model': self.model,
                    'messages': [
                        system_message,
                        init_message
                    ]
                },
                timeout = TIMEOUT
            )

            if add_to_history:
                history.append(system_message)
                history.append(init_message)
        else:
            next_message = {
                'role': 'user',
                'content': prompt
            }

            response = post(
                self.url,
                json = {
                    'model': self.model,
                    'messages': [*history, next_message]
                },
                timeout = TIMEOUT
            )

            if add_to_history:
                history.append(next_message)

        if response.status_code == 200:
            response_message = response.json()['choices'][0]['message']

            response_message.pop('reasoning_content')
            response_message.pop('tool_calls')

            if add_to_history:
                history.append(response_message)

            return response_message['content']

        logger.error('Unexpected response to messages:\n%s\n', dumps(history, indent = 2, ensure_ascii = False))

        raise ValueError(f'Unexpected response from model {self.label} ({self.model}): {response.text}')
