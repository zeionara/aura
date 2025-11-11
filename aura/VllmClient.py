from requests import post


TIMEOUT = 3600


class VllmClient:
    def __init__(self, host: str, port: int, model: str, system_prompt: str):
        self.host = host
        self.port = port
        self.model = model
        self.system_prompt = system_prompt

    @property
    def url(self):
        return f'http://{self.host}:{self.port}/v1/chat/completions'

    def complete(self, prompt: str):
        response = post(
            self.url,
            json = {
                'model': self.model,
                'messages': [
                    {
                        'role': 'system',
                        'content': self.system_prompt
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            },
            timeout = TIMEOUT
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']

        raise ValueError(f'Unexpected response: {response.text}')
