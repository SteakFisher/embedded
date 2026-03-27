const PORT = Number.parseInt(process.env.PORT ?? "3001", 10);
const HOST = process.env.HOST ?? "0.0.0.0";

type LatestImage = {
  bytes: Uint8Array;
  mimeType: string;
  uploadedAt: string;
};

let latestImage: LatestImage | null = null;

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

const jsonResponse = (body: unknown, status = 200) =>
  Response.json(body, {
    status,
    headers: corsHeaders,
  });

const server = Bun.serve({
  port: PORT,
  hostname: HOST,
  async fetch(req) {
    const url = new URL(req.url);

    if (req.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: corsHeaders,
      });
    }

    if (req.method === "POST" && url.pathname === "/upload") {
      const contentType = req.headers.get("content-type") ?? "";
      try {
        if (contentType.includes("multipart/form-data")) {
          const formData = await req.formData();
          const uploaded = formData.get("image");

          if (!(uploaded instanceof File)) {
            return jsonResponse({ error: "Missing file field 'image'." }, 400);
          }

          if (!uploaded.type.startsWith("image/")) {
            return jsonResponse({ error: "Only image files are allowed." }, 400);
          }

          const bytes = new Uint8Array(await uploaded.arrayBuffer());

          latestImage = {
            bytes,
            mimeType: uploaded.type || "application/octet-stream",
            uploadedAt: new Date().toISOString(),
          };
        } else if (contentType.startsWith("image/")) {
          latestImage = {
            bytes: new Uint8Array(await req.arrayBuffer()),
            mimeType: contentType,
            uploadedAt: new Date().toISOString(),
          };
        } else {
          return jsonResponse(
            {
              error:
                "Send multipart/form-data with field 'image' or raw image bytes with content-type image/*.",
            },
            400,
          );
        }

        return jsonResponse({ ok: true, uploadedAt: latestImage.uploadedAt });
      } catch (error) {
        const message = error instanceof Error ? error.message : "Upload failed.";
        return jsonResponse({ error: message }, 500);
      }
    }

    if (req.method === "GET" && url.pathname === "/latest-image") {
      if (!latestImage) {
        return jsonResponse({ error: "No image uploaded yet." }, 404);
      }

      return new Response(latestImage.bytes, {
        status: 200,
        headers: {
          ...corsHeaders,
          "Content-Type": latestImage.mimeType,
          "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
          Pragma: "no-cache",
          Expires: "0",
        },
      });
    }

    if (req.method === "GET" && url.pathname === "/health") {
      return jsonResponse({ ok: true, hasImage: Boolean(latestImage) });
    }

    return jsonResponse({ error: "Not found" }, 404);
  },
});

console.log(`Backend listening on http://${HOST}:${server.port}`);
