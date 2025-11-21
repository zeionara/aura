from enum import Enum
from requests import post, get
from logging import getLogger

from .LLMClient import LLMClient, TIMEOUT


logger = getLogger(__name__)

MAX_AUTHORIZATION_ATTEMPTS = 2


class GigaChatModel(Enum):
    GIGACHAT = 'GigaChat'
    GIGACHAT2 = 'GigaChat-2'
    GIGACHAT2MAX = 'GigaChat-2-Max'
    GIGACHAT2PRO = 'GigaChat-2-Pro'
    GIGACHATMAX = 'GigaChat-Max'
    GIGACHATMAXPREVIEW = 'GigaChat-Max-preview'
    GIGACHATPLUS = 'GigaChat-Plus'
    GIGACHATPRO = 'GigaChat-Pro'
    GIGACHATPROPREVIEW = 'GigaChat-Pro-preview'
    GIGACHATPREVIEW = 'GigaChat-preview'


class GigaChatClient(LLMClient):
    def __init__(self, authorization_key: str, model: GigaChatModel, system_prompt: str, label: str):
        super().__init__(system_prompt, label)

        self.authorization_key = authorization_key
        self.access_token = None
        self.model = model

        self.history = []

    @property
    def url(self):
        return 'https://gigachat.devices.sberbank.ru/api/v1/chat/completions'

    @property
    def authorization_url(self):
        return "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    @property
    def models_url(self):
        return 'https://gigachat.devices.sberbank.ru/api/v1/models'

    def reset(self):
        self.history = []

    def _authorize(self, n_attempts: int = 1):
        response = post(
            self.authorization_url,
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json',
                'RqUID': '13cfdd45-0f5b-4786-8ff6-34d8e85d6354',
                'Authorization': f'Basic {self.authorization_key}'
            },
            data = {
                'scope': 'GIGACHAT_API_PERS'
            },
            timeout = TIMEOUT,
            verify = False
        )

        if response.status_code == 200:
            self.access_token = response.json().get('access_token')
            return True
        else:
            print(n_attempts)
            if n_attempts < MAX_AUTHORIZATION_ATTEMPTS:
                return self._authorize(n_attempts + 1)

        logger.error('Unexpected response status code from model %s authorization service: %d (%s)', self.label, response.status_code, response.text)
        return False

    def _complete(self, messages: list, allow_retry: bool = True):
        if self.access_token is None:
            self._authorize()

            if allow_retry:
                allow_retry = False

        response = post(
            self.url,
            json = {
                'model': self.model.value,
                'messages': messages
            },
            headers = {
                'Authorization': f'Bearer {self.access_token}'
            },
            timeout = TIMEOUT,
            verify = False
        )

        if response.status_code == 200:
            response_message = response.json()['choices'][0]['message']

            return response_message

        if response.status_code == 401 and allow_retry:
            self._authorize()
            return self._complete(messages, allow_retry = False)

        logger.error('Unexpected response status code from model %s: %d (%s)', self.label, response.status_code, response.text)
        return None

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

            if add_to_history:
                history.append(system_message)
                history.append(init_message)

            response = self._complete(
                messages = [
                    system_message,
                    init_message
                ]
            )

            if response is None:
                return response
        else:
            next_message = {
                'role': 'user',
                'content': prompt
            }

            response = self._complete(
                messages = [*history, next_message]
            )

            if response is None:
                return response

            if add_to_history:
                history.append(next_message)

        return response.get('content')

    def models(self, allow_retry: bool = True):
        if self.access_token is None:
            self._authorize()

            if allow_retry:
                allow_retry = False

        response = get(
            self.models_url,
            headers = {
                'Authorization': f'Bearer {self.access_token}'
            },
            timeout = TIMEOUT,
            verify = False
        )

        if response.status_code == 200:
            response_message = [item.get('id') for item in response.json()['data']]

            return response_message

        if response.status_code == 401 and allow_retry:
            self._authorize()
            return self.models(allow_retry = False)

        logger.error('Unexpected response status code from model %s when listing models: %d (%s)', self.label, response.status_code, response.text)
        return None
