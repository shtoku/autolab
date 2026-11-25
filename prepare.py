import os
import struct
import time
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FILENAME   = "data/tyranno.3df"  # ~10,000 pts
# FILENAME = "data/merry.3df"      # ~100,000 pts
WIDTH      = 1920
HEIGHT     = 1080
PITCH      = 6.4e-6
WAVELENGTH = 532e-9
DISTANCE   = 0.3
EVAL_DEPTHS = (DISTANCE - 0.003, DISTANCE, DISTANCE + 0.003)


def _make_data(filename: str) -> torch.Tensor:
  with open(filename, "rb") as f:
    (object_size,) = struct.unpack("<i", f.read(4))
    buf = bytearray(f.read())
    object_data = torch.frombuffer(buf, dtype=torch.float32).reshape(-1, 3)

  hologram_size = torch.tensor([WIDTH, HEIGHT])
  max_indices = hologram_size - 1.0

  min_coords, max_coords = object_data.min(dim=0).values, object_data.max(dim=0).values
  range_coords = torch.clamp(max_coords - min_coords, min=1.0)

  object_data = object_data - (min_coords + max_coords) / 2.0
  object_data = object_data * torch.min(max_indices / range_coords[:2])
  object_data = PITCH * object_data + torch.tensor([0.0, 0.0, DISTANCE])

  return object_data


def _exact_field(object_data: torch.Tensor, width: int, height: int, pitch: float, wavelength: float) -> torch.Tensor:
  device = object_data.device
  hx = pitch * torch.arange(-width // 2, width // 2, device=device, dtype=torch.float32)
  hy = pitch * torch.arange(-height // 2, height // 2, device=device, dtype=torch.float32)
  gx, gy = torch.meshgrid(hx, hy, indexing="xy")

  k = 2.0 * torch.pi / wavelength
  field = torch.zeros((height, width), dtype=torch.complex64, device=device)
  for i in range(0, object_data.shape[0], 64):
    p = object_data[i:i + 64]
    dx = gx.unsqueeze(0) - p[:, 0].view(-1, 1, 1)
    dy = gy.unsqueeze(0) - p[:, 1].view(-1, 1, 1)
    r = torch.sqrt(dx * dx + dy * dy + p[:, 2].view(-1, 1, 1) ** 2)
    field += torch.sum(torch.polar(1.0 / r, k * r), dim=0)
  return field


def _propagate(u: torch.Tensor, z: float, pitch: float = PITCH, wavelength: float = WAVELENGTH) -> torch.Tensor:
  h, w = u.shape
  device = u.device
  u = torch.fft.fftshift(u, dim=(-2, -1))
  u = torch.fft.fft2(u)
  u = torch.fft.fftshift(u, dim=(-2, -1))

  x = torch.arange(-w // 2, w // 2, device=device, dtype=torch.float32)
  y = torch.arange(-h // 2, h // 2, device=device, dtype=torch.float32)
  X, Y = torch.meshgrid(x, y, indexing="xy")
  fx, fy = X / (w * pitch), Y / (h * pitch)
  phase = z * torch.sqrt(torch.clamp(1.0 / wavelength**2 - fx**2 - fy**2, min=0.0))

  h_transfer = torch.exp(1j * 2.0 * torch.pi * phase)
  u = u * h_transfer

  u = torch.fft.fftshift(u, dim=(-2, -1))
  u = torch.fft.ifft2(u)
  u = torch.fft.fftshift(u, dim=(-2, -1))

  return u


def _reconstruct_amp(holo: torch.Tensor, z: float = DISTANCE) -> torch.Tensor:
  rec = _propagate(holo, -z)
  return torch.abs(rec)


def _sim(amp_a: torch.Tensor, amp_b: torch.Tensor) -> float:
  scale = amp_b.max() * 0.5 + 1e-8
  a = torch.clamp(amp_a / scale, 0.0, 1.0)
  b = torch.clamp(amp_b / scale, 0.0, 1.0)

  a = F.avg_pool2d(a.unsqueeze(0).unsqueeze(0), kernel_size=4, stride=4).flatten()
  b = F.avg_pool2d(b.unsqueeze(0).unsqueeze(0), kernel_size=4, stride=4).flatten()
  a = a - a.mean()
  b = b - b.mean()
  return F.cosine_similarity(a, b, dim=0).item()


OBJECT_DATA = _make_data(FILENAME).to(DEVICE)
_REF_AMPS = None


def _get_ref_amps() -> list[torch.Tensor]:
  global _REF_AMPS
  if _REF_AMPS is None:
    stem = os.path.splitext(os.path.basename(FILENAME))[0]
    cache_path = f"_ref_amps_{stem}_N{OBJECT_DATA.shape[0]}.pt"
    if os.path.exists(cache_path):
      _REF_AMPS = torch.load(cache_path, map_location=DEVICE)
    else:
      ref_holo = _exact_field(OBJECT_DATA, WIDTH, HEIGHT, PITCH, WAVELENGTH)
      _REF_AMPS = [_reconstruct_amp(ref_holo, z) for z in EVAL_DEPTHS]
      torch.save(_REF_AMPS, cache_path)
  return _REF_AMPS


def dry_run(fn) -> None:
  with torch.no_grad():
    # N = 0 のエッジケース検証
    out_empty = fn(OBJECT_DATA[:0], WIDTH, HEIGHT, PITCH, WAVELENGTH)
    assert out_empty.shape == (HEIGHT, WIDTH) and torch.is_complex(out_empty), "Failed on N=0"

    # N = 64 での精度・型検証
    pts_small = OBJECT_DATA[:64]
    out = fn(pts_small, WIDTH, HEIGHT, PITCH, WAVELENGTH)
    assert isinstance(out, torch.Tensor) and out.shape == (HEIGHT, WIDTH), f"Bad shape: {getattr(out, 'shape', type(out))}"
    assert torch.is_complex(out), f"Expected complex tensor, got {out.dtype}"
    assert out.device == pts_small.device, f"Bad device: {out.device}"

    ref_small = _exact_field(pts_small, WIDTH, HEIGHT, PITCH, WAVELENGTH)
    sim = _sim(_reconstruct_amp(out, DISTANCE), _reconstruct_amp(ref_small, DISTANCE))
    assert sim > 0.85, f"Low reconstructed image accuracy: sim={sim:.3f}"

    # 実データおよび 100,000点スケールでのメモリ・動作検証
    _ = fn(OBJECT_DATA, WIDTH, HEIGHT, PITCH, WAVELENGTH)
    if OBJECT_DATA.shape[0] < 100_000:
      pts_100k = OBJECT_DATA.repeat((100_000 // OBJECT_DATA.shape[0]) + 1, 1)[:100_000]
      _ = fn(pts_100k, WIDTH, HEIGHT, PITCH, WAVELENGTH)

  if DEVICE == "cuda":
    torch.cuda.synchronize()


def evaluate(fn, warmup: int = 1, iters: int = 3) -> dict:
  try:
    with torch.no_grad():
      for _ in range(warmup):
        out = fn(OBJECT_DATA, WIDTH, HEIGHT, PITCH, WAVELENGTH)
      if DEVICE == "cuda":
        torch.cuda.synchronize()

      if not isinstance(out, torch.Tensor) or out.shape != (HEIGHT, WIDTH) or not torch.is_complex(out):
        return {"valid": False, "error": "Invalid shape or non-complex dtype"}
      if not torch.isfinite(out).all():
        return {"valid": False, "error": "NaN or Inf"}

      ref_amps = _get_ref_amps()
      sims = [_sim(_reconstruct_amp(out, z), ref_amp) for z, ref_amp in zip(EVAL_DEPTHS, ref_amps)]
      sim = min(sims)
      if sim < 0.82:
        return {"valid": False, "error": f"rec_sim={sim:.4f} < 0.82 (depths={[round(s, 3) for s in sims]})"}

      if DEVICE == "cuda":
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
          fn(OBJECT_DATA, WIDTH, HEIGHT, PITCH, WAVELENGTH)
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / iters
      else:
        t0 = time.perf_counter()
        for _ in range(iters):
          fn(OBJECT_DATA, WIDTH, HEIGHT, PITCH, WAVELENGTH)
        ms = (time.perf_counter() - t0) * 1000.0 / iters

      return {"valid": True, "time_ms": round(ms, 3), "sim": round(sim, 4), "num_points": int(OBJECT_DATA.shape[0])}
  except Exception as e:
    return {"valid": False, "error": f"{type(e).__name__}: {e}"}


if __name__ == '__main__':
  import cv2
  import train

  with torch.no_grad():
    reference = _get_ref_amps()[1]
    target = train.function(OBJECT_DATA, WIDTH, HEIGHT, PITCH, WAVELENGTH)
    target = _reconstruct_amp(target, DISTANCE)

    similarity = _sim(reference, target)
    print(f"File={FILENAME}, N={OBJECT_DATA.shape[0]}, Similarity: {similarity:.4f}")

    o = target ** 2
    o = 3.0 * (255.0 * o / o.max())
    o = torch.clamp(o, 0, 255).to(torch.uint8)
    cv2.imwrite('o.png', o.cpu().numpy())