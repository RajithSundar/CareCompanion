# Frontend build

`static/app.css` is built from `input.css` with the Tailwind CSS v3.4.17 standalone CLI.
The built file is committed, so the server needs no Node toolchain.

Rebuild after changing classes in `static/index.html` or `static/app.js`:

```bash
# one-time: download the CLI (Linux x64) and check its SHA-256
curl -sSLf -o frontend/bin/tailwindcss \
  https://github.com/tailwindlabs/tailwindcss/releases/download/v3.4.17/tailwindcss-linux-x64
echo "7d24f7fa191d2193b78cd5f5a42a6093e14409521908529f42d80b11fde1f1d4  frontend/bin/tailwindcss" | sha256sum -c -
chmod +x frontend/bin/tailwindcss

cd frontend && ./bin/tailwindcss -c tailwind.config.js -i input.css -o ../static/app.css --minify
```

`static/vendor/` holds Chart.js 4.4.1 and the Lucide 0.468.0 licence (the icons the site uses live in `static/icons.js`, extracted from the Lucide UMD build from the npm
registry, verified against the registry shasum). Their licenses sit next to them.
