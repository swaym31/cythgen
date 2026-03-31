---
title: CythGen
emoji: 🔤
colorFrom: gray
colorTo: black
sdk: docker
pinned: false
---

# CythGen — Cythoran Glyph Generator

FastAPI backend for generating Cythoran script glyphs from English names.

## API

**POST /generate**
```json
{ "name": "tanmay", "threshold": 0.5 }
```
Returns:
```json
{
  "name": "tanmay",
  "normalized": "tanmai",
  "chunks": ["tan", "mai"],
  "glyphs": ["<base64 PNG>", "<base64 PNG>"]
}
```

**GET /health** — check if model is loaded
