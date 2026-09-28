import argparse
import io
from http.server import BaseHTTPRequestHandler, HTTPServer

from diffusers.utils import load_image

PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AnyBokeh Focus Picker</title>
<style>
  body { font-family: sans-serif; margin: 16px; }
  #wrap { position: relative; display: inline-block; }
  #img { max-width: 95vw; max-height: 80vh; cursor: crosshair; display: block; }
  #dot { position: absolute; width: 12px; height: 12px; margin: -8px 0 0 -8px; border: 2px solid red;
         border-radius: 50%; pointer-events: none; display: none; }
  #out { font-family: monospace; font-size: 18px; margin: 12px 0; }
</style>
</head>
<body>
<div id="out">Click on the image to pick the focus point (__SIZE__).</div>
<div id="wrap"><img id="img" src="/image.jpg"><div id="dot"></div></div>
<script>
  const img = document.getElementById("img"), dot = document.getElementById("dot");
  img.addEventListener("click", (e) => {
    const x = Math.min(Math.floor(e.offsetX * img.naturalWidth / img.clientWidth), img.naturalWidth - 1);
    const y = Math.min(Math.floor(e.offsetY * img.naturalHeight / img.clientHeight), img.naturalHeight - 1);
    document.getElementById("out").textContent = `--focus_x ${x} --focus_y ${y}`;
    dot.style.left = e.offsetX + "px";
    dot.style.top = e.offsetY + "px";
    dot.style.display = "block";
  });
</script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Pick the focus point of an image in the browser.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main(args):
    # Same loader as inference (applies the EXIF orientation), so the picked coordinates match.
    image = load_image(args.image_path)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    image_bytes = buffer.getvalue()
    page_bytes = PAGE.replace("__SIZE__", f"{image.width}x{image.height}").encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                body, content_type = page_bytes, "text/html; charset=utf-8"
            elif self.path == "/image.jpg":
                body, content_type = image_bytes, "image/jpeg"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = HTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port} in your browser and click on the image. Press Ctrl+C to quit.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
