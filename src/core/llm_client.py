from groq import Groq, RateLimitError, AuthenticationError, APIError

from src.core.config import settings


class LLMClient:
    """
    Thin wrapper around Groq chat completions. Every other module
    (generation, query rewriting, critic, citation checks) talks to
    this interface, not to Groq directly — swapping providers means
    editing only this file.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._client = Groq(api_key=api_key or settings.groq_api_key)
        self.model = model or settings.groq_model

    def complete(
        self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.2, model: str | None = None
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            raise RuntimeError("Groq API rate limit reached. Please try again shortly.")
        except AuthenticationError:
            raise RuntimeError("Invalid Groq API key. Check your .env file.")
        except APIError as e:
            raise RuntimeError(f"Groq API error: {e}")
