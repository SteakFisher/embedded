# backend

Simple Bun backend that always keeps only the latest uploaded image in memory.

## Setup

```bash
bun install
```

Copy env file:

```bash
cp .env.example .env
```

## Run

```bash
bun run dev
```

or

```bash
bun run start
```

## Env

- `PORT` (default `3001`)
- `HOST` (default `0.0.0.0`)

## API

- `POST /upload`
  - Accepts either:
    - `multipart/form-data` with field `image`, or
    - raw image bytes with `Content-Type: image/*`
  - Replaces the previous image in memory.

- `GET /latest-image`
  - Returns the latest image bytes.
  - Returns `404` if no image has been uploaded yet.

- `GET /health`
  - Returns backend status and whether an image exists.

## Quick upload examples

Multipart:

```bash
curl -X POST http://localhost:3001/upload -F "image=@plant.jpg"
```

Raw bytes:

```bash
curl -X POST http://localhost:3001/upload --data-binary "@plant.jpg" -H "Content-Type: image/jpeg"
```
