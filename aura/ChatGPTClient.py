from requests import post


URL = 'https://api.openai.com/v1/chat/completions'
TIMEOUT = 3600


class ChatGPTClient:
    def __init__(self, token: str, system_prompt: str):
        self.token = token
        self.system_prompt = system_prompt

    def ask(self, prompt: str):
        response = post(
            URL,
            headers = {
                'Authorization': f'Bearer {self.token}'
            },
            json = {
                'model': 'gpt-4o',
                'messages': [
                    {
                        'role': 'developer',
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
