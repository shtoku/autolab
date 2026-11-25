import torch

def function(object_data: torch.Tensor, width: int, height: int, pitch: float, wavelength: float) -> torch.Tensor:
  device = object_data.device
  field = torch.zeros((height, width), dtype=torch.complex64, device=device)

  hx = pitch * torch.arange(-width // 2, width // 2, device=device, dtype=torch.float32)
  hy = pitch * torch.arange(-height // 2, height // 2, device=device, dtype=torch.float32)
  gx, gy = torch.meshgrid(hx, hy, indexing="xy")

  k = 2.0 * torch.pi / wavelength
  for i in range(0, object_data.shape[0]):
    dx = gx - object_data[i, 0]
    dy = gy - object_data[i, 1]
    dz = object_data[i, 2]
    r = torch.sqrt(dx * dx + dy * dy + dz * dz)
    field += torch.exp(1j * k * r) / r

  return field
