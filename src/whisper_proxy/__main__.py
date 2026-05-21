import uvicorn


def main() -> None:
    uvicorn.run(
        "whisper_proxy.app:app",
        host="0.0.0.0",
        port=9000,
        loop="uvloop",
        http="httptools",
        access_log=False,
    )


if __name__ == "__main__":
    main()
