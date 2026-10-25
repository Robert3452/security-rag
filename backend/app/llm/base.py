from typing import Protocol


class LLMError(Exception):
    """El proveedor del LLM falló: red, límites de uso o autenticación."""


class LLM(Protocol):
    async def generate(self, system: str, user: str) -> str: ...
