from typing import cast

from fastapi import Request

from whisper_proxy.config import Settings
from whisper_proxy.openarc import OpenArcClient


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_openarc_client(request: Request) -> OpenArcClient:
    return cast(OpenArcClient, request.app.state.openarc_client)
