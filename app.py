import os, io, base64, gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# Auto-download model weights from Google Drive if not present
# ─────────────────────────────────────────────────────────────────────────────

GDRIVE_ID = "10ophvoiEuPtqAszNEIU5pVfqRuvalRf4"
CKPT_PATH = "best.pt"

if not os.path.exists(CKPT_PATH):
    print("Downloading best.pt from Google Drive...")
    gdown.download(id=GDRIVE_ID, output=CKPT_PATH, quiet=False)
    print("Download complete.")

# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_TYPE_TO_ID = {"bases": 0, "1b1s": 1, "1b2s": 2}

def letter_id(ch):
    if ch is None: return -1
    ch = str(ch).strip().lower()
    if len(ch) != 1: return -1
    if "a" <= ch <= "z": return ord(ch) - ord("a")
    return -1

class ResBlock(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
    def forward(self, x): return x + self.net(x)

class FiLM(nn.Module):
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    def forward(self, x, cond_vec):
        gamma, beta = self.proj(cond_vec).chunk(2, dim=1)
        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta  = beta.view(-1,  x.size(1), 1, 1)
        return (1 + gamma) * x + beta

class SelfAttention2d(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H*W).permute(0, 2, 1).float()
        h, _ = self.attn(h, h, h)
        return x + h.to(x.dtype).permute(0, 2, 1).view(B, C, H, W)

class ConditionEmbed(nn.Module):
    def __init__(self, emb_dim=32, cond_dim=256):
        super().__init__()
        self.none_id   = 26
        self.base_emb  = nn.Embedding(26, emb_dim)
        self.sound_emb = nn.Embedding(27, emb_dim)
        self.type_emb  = nn.Embedding(3,  emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim * 4, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),    nn.SiLU(),
        )
    def forward(self, base_id, s1_id, s2_id, type_id):
        s1_id = torch.where(s1_id < 0, torch.full_like(s1_id, self.none_id), s1_id)
        s2_id = torch.where(s2_id < 0, torch.full_like(s2_id, self.none_id), s2_id)
        b  = self.base_emb( base_id.clamp(0, 25))
        s1 = self.sound_emb(s1_id.clamp(0, 26))
        s2 = self.sound_emb(s2_id.clamp(0, 26))
        t  = self.type_emb( type_id.clamp(0, 2))
        return self.proj(torch.cat([b, s1, s2, t], dim=1))

class ResEncoder(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=256, ch=64):
        super().__init__()
        self.stem  = nn.Conv2d(1, ch, 3, padding=1)
        self.down1 = nn.Sequential(ResBlock(ch),   nn.Conv2d(ch,   ch*2, 4, 2, 1))
        self.film1 = FiLM(cond_dim, ch*2)
        self.down2 = nn.Sequential(ResBlock(ch*2), nn.Conv2d(ch*2, ch*4, 4, 2, 1))
        self.film2 = FiLM(cond_dim, ch*4)
        self.down3 = nn.Sequential(ResBlock(ch*4), nn.Conv2d(ch*4, ch*8, 4, 2, 1))
        self.film3 = FiLM(cond_dim, ch*8)
        self.down4 = nn.Sequential(ResBlock(ch*8), nn.Conv2d(ch*8, ch*8, 4, 2, 1))
        self.film4 = FiLM(cond_dim, ch*8)
        self.attn    = SelfAttention2d(ch*8)
        self.res_mid = ResBlock(ch*8)
        feat_dim = ch * 8 * 16 * 16
        self.fc  = nn.Sequential(nn.Linear(feat_dim + cond_dim, 512), nn.SiLU())
        self.mu     = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)
    def forward(self, x, cond_vec):
        h = self.stem(x)
        h = self.down1(h); h = self.film1(h, cond_vec)
        h = self.down2(h); h = self.film2(h, cond_vec)
        h = self.down3(h); h = self.film3(h, cond_vec)
        h = self.down4(h); h = self.film4(h, cond_vec)
        h = self.attn(h); h = self.res_mid(h)
        h = h.flatten(1)
        h = self.fc(torch.cat([h, cond_vec], dim=1))
        return self.mu(h), self.logvar(h)

class ResDecoder(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=256, ch=64):
        super().__init__()
        self.ch  = ch
        feat_dim = ch * 8 * 16 * 16
        self.fc  = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 512), nn.SiLU(),
            nn.Linear(512, feat_dim),              nn.SiLU(),
        )
        self.res_mid = ResBlock(ch*8)
        self.attn    = SelfAttention2d(ch*8)
        self.up4  = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch*8, ch*8, 3, padding=1))
        self.film4 = FiLM(cond_dim, ch*8); self.res4 = ResBlock(ch*8)
        self.up3  = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch*8, ch*4, 3, padding=1))
        self.film3 = FiLM(cond_dim, ch*4); self.res3 = ResBlock(ch*4)
        self.up2  = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch*4, ch*2, 3, padding=1))
        self.film2 = FiLM(cond_dim, ch*2); self.res2 = ResBlock(ch*2)
        self.up1  = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch*2, ch,   3, padding=1))
        self.film1 = FiLM(cond_dim, ch);   self.res1 = ResBlock(ch)
        self.out  = nn.Conv2d(ch, 1, 1)
    def forward(self, z, cond_vec):
        h = self.fc(torch.cat([z, cond_vec], dim=1))
        h = h.view(h.size(0), self.ch * 8, 16, 16)
        h = self.res_mid(h); h = self.attn(h)
        h = self.up4(h); h = self.film4(h, cond_vec); h = self.res4(h)
        h = self.up3(h); h = self.film3(h, cond_vec); h = self.res3(h)
        h = self.up2(h); h = self.film2(h, cond_vec); h = self.res2(h)
        h = self.up1(h); h = self.film1(h, cond_vec); h = self.res1(h)
        return self.out(h)

class ResCVAE(nn.Module):
    def __init__(self, latent_dim=128, emb_dim=32, cond_dim=256, ch=64):
        super().__init__()
        self.cond_net   = ConditionEmbed(emb_dim=emb_dim, cond_dim=cond_dim)
        self.enc        = ResEncoder(latent_dim=latent_dim, cond_dim=cond_dim, ch=ch)
        self.dec        = ResDecoder(latent_dim=latent_dim, cond_dim=cond_dim, ch=ch)
        self.latent_dim = latent_dim
    def forward(self, x, base_id, s1_id, s2_id, type_id):
        c = self.cond_net(base_id, s1_id, s2_id, type_id)
        mu, logvar = self.enc(x, c)
        logvar = logvar.clamp(-10, 10)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.dec(z, c), mu, logvar

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cpu")
model  = ResCVAE(latent_dim=128, emb_dim=32, cond_dim=256, ch=64).to(device)

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print("Model loaded successfully.")
else:
    print("WARNING: best.pt not found.")

model.eval()

# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

LETTER_MAP = {"c":"s","q":"k","x":"k","z":"j","w":"v","y":"i","u":"o"}

def normalize(text):
    text = "".join(ch.lower() for ch in str(text) if ch.isalpha())
    word = "".join(LETTER_MAP.get(ch, ch) for ch in text)
    if word.startswith("h"): word = word[1:]
    return word

def chunk_word(s):
    n = len(s)
    memo = {}
    def dfs(i):
        if i == n: return []
        if i in memo: return memo[i]
        if s[i] == "h": memo[i] = None; return None
        best = None
        for L in (3, 2, 1):
            if i + L > n: continue
            if i + L == n and L == 1 and n > 1: continue
            suf = dfs(i + L)
            if suf is None: continue
            cand = [s[i:i+L]] + suf
            if best is None: best = cand; continue
            cl = [len(x) for x in cand]
            bl = [len(x) for x in best]
            for k in range(min(len(cl), len(bl))):
                if cl[k] > bl[k]: best = cand; break
                if cl[k] < bl[k]: break
        memo[i] = best; return best
    return dfs(0) or []

def parse_chunk(ch):
    ch = "".join(c for c in ch.lower() if c.isalpha())
    if len(ch) == 1: return ch[0], None, None, "bases"
    if len(ch) == 2: return ch[0], ch[1], None, "1b1s"
    if len(ch) == 3: return ch[0], ch[1], ch[2], "1b2s"
    raise ValueError(f"Invalid chunk: {ch}")

# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

NONE_ID = 26

def make_cond(base, s1=None, s2=None, chunk_type="1b2s"):
    base_id = letter_id(base)
    s1_id   = letter_id(s1) if s1 else -1
    s2_id   = letter_id(s2) if s2 else -1
    type_id = CHUNK_TYPE_TO_ID.get(chunk_type, 2)
    s1_id   = NONE_ID if s1_id < 0 else s1_id
    s2_id   = NONE_ID if s2_id < 0 else s2_id
    to_t    = lambda v: torch.tensor([v], dtype=torch.long, device=device)
    return to_t(base_id), to_t(s1_id), to_t(s2_id), to_t(type_id)

def tensor_to_b64(t):
    arr = (t.numpy() * 255).astype("uint8")
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

@torch.no_grad()
def generate_chunk(chunk_str, threshold=0.5):
    base, s1, s2, chunk_type = parse_chunk(chunk_str)
    base_t, s1_t, s2_t, type_t = make_cond(base, s1, s2, chunk_type)
    c    = model.cond_net(base_t, s1_t, s2_t, type_t)
    z    = torch.zeros(1, model.latent_dim, device=device)
    xhat = model.dec(z, c)
    img  = (torch.sigmoid(xhat).cpu() > threshold).float()
    return img[0, 0]

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    name: str
    threshold: float = 0.5

@app.get("/")
def root():
    return {"status": "CythGen API running"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(CKPT_PATH)}

@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        norm = normalize(req.name)
        if not norm:
            return {"error": "Empty after normalization", "glyphs": [], "chunks": []}
        chunks = chunk_word(norm)
        if not chunks:
            return {"error": "Could not chunk name", "glyphs": [], "chunks": []}
        glyphs = []
        for ch in chunks:
            try:
                glyphs.append(tensor_to_b64(generate_chunk(ch, req.threshold)))
            except Exception as e:
                print(f"Chunk '{ch}' failed: {e}")
                glyphs.append(tensor_to_b64(torch.ones(256, 256)))
        return {"name": req.name, "normalized": norm, "chunks": chunks, "glyphs": glyphs}
    except Exception as e:
        return {"error": str(e), "glyphs": [], "chunks": []}
