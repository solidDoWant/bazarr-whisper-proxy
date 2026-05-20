from typing import cast

from fastapi import Request

from whisper_proxy.config import Settings
from whisper_proxy.lingarr import LingarrClient
from whisper_proxy.openarc import OpenArcClient


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_openarc_client(request: Request) -> OpenArcClient:
    return cast(OpenArcClient, request.app.state.openarc_client)


def get_lingarr_client(request: Request) -> LingarrClient | None:
    return cast(LingarrClient | None, request.app.state.lingarr_client)
