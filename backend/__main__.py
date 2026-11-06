"""python -m backend [--host 127.0.0.1] [--port 8000] [--device auto|cpu|cuda] [--weights path]"""
import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description="Photonika: веб-сервер (API + фронтенд)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--reload", action="store_true", help="перезапуск при изменении кода (разработка)")
    args = ap.parse_args()
    os.environ["PHOTONIKA_DEVICE"] = args.device
    if args.weights:
        os.environ["PHOTONIKA_WEIGHTS"] = args.weights
    import uvicorn

    print(f"Photonika: http://{args.host}:{args.port}  (документация API: /api/docs)", flush=True)
    uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
