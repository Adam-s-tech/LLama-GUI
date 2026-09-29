const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const UI_ROOT = path.resolve(__dirname, "..", "..", "ui");

function contentType(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    return {
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".ico": "image/x-icon",
    }[ext] || "application/octet-stream";
}

// Reuse connections to avoid Windows socket pressure (net::ERR_NO_BUFFER_SPACE)
// during repeated loads. Python's default HTTP/1.0 server closed after each asset.
function startUiServer(port = 0) {
    const server = http.createServer((request, response) => {
        if (request.method !== "GET" && request.method !== "HEAD") {
            response.writeHead(405, { Allow: "GET, HEAD" });
            response.end();
            return;
        }
        let pathname;
        try {
            pathname = decodeURIComponent(new URL(request.url, "http://127.0.0.1").pathname);
        } catch (error) {
            response.writeHead(400);
            response.end("Bad request");
            return;
        }
        const relative = pathname === "/" ? "index.html" : pathname.replace(/^\/+/, "");
        const filePath = path.resolve(UI_ROOT, relative);
        if (filePath !== UI_ROOT && !filePath.startsWith(UI_ROOT + path.sep)) {
            response.writeHead(403);
            response.end("Forbidden");
            return;
        }
        if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
            response.writeHead(404);
            response.end("Not found");
            return;
        }
        response.writeHead(200, {
            "Content-Type": contentType(filePath),
            "Cache-Control": "no-store",
        });
        if (request.method === "HEAD") response.end();
        else response.end(fs.readFileSync(filePath));
    });
    return new Promise((resolve, reject) => {
        const onError = error => reject(error);
        server.once("error", onError);
        server.listen(port, "127.0.0.1", () => {
            server.removeListener("error", onError);
            const address = server.address();
            resolve({
                baseUrl: `http://127.0.0.1:${address.port}/`,
                async close() {
                    if (typeof server.closeAllConnections === "function") server.closeAllConnections();
                    await new Promise((done, fail) => server.close(error => error ? fail(error) : done()));
                },
            });
        });
    });
}

module.exports = { startUiServer };
