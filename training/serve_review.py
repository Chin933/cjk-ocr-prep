"""Serve local review assets with process-owned log handles on Windows."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    log_dir = root.parent / 'runs' / 'note_batch_check'
    log_dir.mkdir(parents=True, exist_ok=True)
    # The server keeps these handles itself after the launching shell exits.
    with (log_dir / 'review-service.log').open('a', encoding='utf-8', buffering=1) as log:
        sys.stdout = sys.stderr = log
        handler = partial(SimpleHTTPRequestHandler, directory=str(root))
        try:
            with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
                print(f'Review server listening on http://127.0.0.1:{args.port}', flush=True)
                server.serve_forever()
        except Exception:
            import traceback
            traceback.print_exc()
            raise
