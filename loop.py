import os
import re
import ast
import json
import time
import httpx
import subprocess
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

MAX_STEPS = 5
HISTORY_FILE = "history.md"

LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:8080")
LLAMA_OPENAI_URL = f"{LLAMA_BASE_URL}/v1"

PLANNER_MODEL = os.getenv("PLANNER_MODEL", "qwen3.8-flash-next")
CODER_MODEL = os.getenv("CODER_MODEL", "qwen3.8-flash-next")
KEEP_MODELS_LOADED = os.getenv("KEEP_MODELS_LOADED", "0") == "1"

PLANNER_CFG = {
  "model": PLANNER_MODEL,
  "temperature": 1.0,
  "top_p": 0.95,
  "presence_penalty": 0.0,
  "max_tokens": 4096,
  "reasoning_budget_tokens": 1500,
  "reasoning_budget_message": "\nVerified math and 16GB VRAM limit up to N=100,000. Outputting the 2-line answer now.",
  "chat_template_kwargs": {
    "enable_thinking": True,
    "reasoning_effort": "medium",
  },
}

CODER_CFG = {
  "model": CODER_MODEL,
  "temperature": 0.2,
  "top_p": 0.80,
  "presence_penalty": 0.0,
  "max_tokens": 4096,
  "chat_template_kwargs": {
    "enable_thinking": False,
  },
}

COORD_SPEC = """device = object_data.device
hologram_x = pitch * torch.arange(-width // 2, width // 2, device=device, dtype=torch.float32)  # (W,)
hologram_y = pitch * torch.arange(-height // 2, height // 2, device=device, dtype=torch.float32)  # (H,)
k = 2.0 * torch.pi / wavelength  # spherical wave phase convention: (1/r) * exp(+1j * k * r)"""


def unload_model(model_name: str) -> None:
  if KEEP_MODELS_LOADED:
    return
  try:
    httpx.post(f"{LLAMA_BASE_URL}/models/unload", json={"model": model_name}, timeout=15.0).raise_for_status()
  except Exception:
    pass


def invoke_llm(cfg: dict, prompt: str, unload_after: bool = True) -> str:
  payload = {**cfg, "messages": [{"role": "user", "content": prompt}]}
  try:
    resp = httpx.post(f"{LLAMA_OPENAI_URL}/chat/completions", json=payload, timeout=300.0)
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]

    txt = (msg.get("content") or "").strip()
    txt = re.sub(r"<think>.*?</think>|<\|thought\|>.*?<\|/thought\|>", "", txt, flags=re.DOTALL)
    if "<|channel|>final<|message|>" in txt:
      txt = txt.split("<|channel|>final<|message|>")[-1]
    txt = txt.strip()

    if not txt:
      rc = (msg.get("reasoning_content") or "").strip()
      txt = ("Approach:" + rc.split("Approach:")[-1]) if "Approach:" in rc else "\n".join(rc.splitlines()[-2:])
    return txt.strip()
  except httpx.HTTPError as e:
    print(f"  [Warning] LLM request error: {e}")
    return ""
  finally:
    if unload_after:
      unload_model(cfg["model"])


class State(TypedDict):
  best_code: str
  best_ms: float
  num_points: int
  idea: str
  code: str
  feedback: str
  prev_feedback: str
  retries: int
  history: list[str]
  step: int


def run_cmd(cmd: str, timeout: int = 180) -> tuple[int, str, str]:
  try:
    p = subprocess.run(["python3", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr
  except subprocess.TimeoutExpired:
    return 1, "", f"Timeout ({timeout}s)"


def log_history(line: str) -> None:
  with open(HISTORY_FILE, "a", encoding="utf-8") as f:
    f.write(f"- {line}\n")


def planner(state: State) -> dict:
  step = state["step"] + 1
  hist = "\n".join(state["history"][-15:]) or "None"

  prompt = (
    f"Current best (N={state['num_points']:,}): {state['best_ms']}ms\n"
    f"```python\n{state['best_code']}\n```\n\n"
    f"History:\n{hist}\n\n"
    "Problem: 3D point cloud holography up to N = 100,000 points (z ~ 0.3m), grid (H, W) = (1080, 1920), pitch = 6.4e-6m, wavelength = 532e-9m.\n"
    "Goal: >5% speedup over current best with reconstructed amplitude similarity >= 0.92 across all 3 depth planes (z = 0.297m, 0.300m, 0.303m).\n"
    "Constraints:\n"
    "- Exact spherical distance `sqrt` is NOT required; any mathematical approximation (e.g., paraxial/Fresnel) that achieves similarity >= 0.92 is valid.\n"
    "- Pure PyTorch eager mode on a dedicated 16GB VRAM GPU (no custom CUDA extensions, no torch.compile).\n"
    "- Scale cleanly up to N = 100,000 points within 16GB VRAM and reduce complexity below O(N*H*W).\n"
    "- Propose an approach structurally distinct from History.\n\n"
    "Output STRICTLY 2 lines:\n"
    "Approach: <short name>\n"
    "Method: <1 sentence describing the math, tensor shapes, and complexity>"
  )

  res = invoke_llm(PLANNER_CFG, prompt, unload_after=(PLANNER_MODEL != CODER_MODEL))
  res = re.sub(r"<think>.*?</think>|```.*?```", "", res, flags=re.DOTALL).replace("**", "")
  lines = [l.strip() for l in res.splitlines() if l.strip()]
  idea = " ".join(lines[-2:])[:350] if lines else "Fallback: empty planner output"

  print(f"\n[{step:02d}/{MAX_STEPS}] {idea}")
  return {"idea": idea, "feedback": "", "prev_feedback": "", "retries": 0, "step": step}


def coder(state: State) -> dict:
  err_ctx = (
    f"\nPrevious attempt:\n```python\n{state['code']}\n```\nError: {state['feedback']}\n"
    "Fix the error while strictly implementing the proposed algorithm.\n"
    if state["feedback"] else ""
  )

  prompt = (
    f"Implement this algorithm from scratch into `function`:\n{state['idea']}\n{err_ctx}\n"
    f"Coordinate & phase conventions:\n```python\n{COORD_SPEC}\n```\n\n"
    "Rules:\n"
    "- Write a clean, concise implementation of the proposed algorithm (avoid verbose comments).\n"
    "- Exact signature: `function(object_data: torch.Tensor, width: int, height: int, pitch: float, wavelength: float) -> torch.Tensor`.\n"
    "- Return complex `(height, width)` tensor on `object_data.device` (handle N = 0 up to N = 100,000 within 16GB VRAM).\n"
    "- Pure PyTorch only (no numpy, scipy, or torch.compile).\n"
    "Output ONLY Python code inside ```python ... ```."
  )

  txt = invoke_llm(CODER_CFG, prompt, unload_after=True)
  txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL)
  m = re.search(r"```(?:python)?\s*\n(.*?)\s*```", txt, re.DOTALL)
  code = m.group(1).strip() if m else txt.strip()
  return {"code": code, "retries": state["retries"] + 1}


def reviewer(state: State) -> dict:
  code = state["code"]
  prev_fb = state["feedback"]

  if not code.strip():
    print("  [NG] Empty code output")
    return {"feedback": "Empty code output", "prev_feedback": prev_fb}

  if code.strip() == state["best_code"].strip():
    print("  [NG] Identical to base code")
    return {"feedback": "Identical to base code", "prev_feedback": prev_fb}

  try:
    ast.parse(code)
  except SyntaxError as e:
    msg = f"SyntaxError line {e.lineno}: {e.msg}"
    print(f"  [NG] {msg}")
    return {"feedback": msg, "prev_feedback": prev_fb}

  if "def function" not in code:
    print("  [NG] Missing `def function`")
    return {"feedback": "Missing `def function`", "prev_feedback": prev_fb}

  with open("_cand.py", "w", encoding="utf-8") as f:
    f.write(code + "\n")
  rc, _, stderr = run_cmd("import prepare, _cand; prepare.dry_run(_cand.function)", timeout=120)
  if os.path.exists("_cand.py"):
    os.remove("_cand.py")

  if rc != 0:
    lines = [l.strip() for l in stderr.splitlines() if l.strip()]
    msg = lines[-1] if lines else "Dry-run failed"
    print(f"  [NG] {msg}")
    return {"feedback": msg, "prev_feedback": prev_fb}

  print("  [OK]")
  return {"feedback": "OK", "prev_feedback": prev_fb}


def evaluator(state: State) -> dict:
  summary = state["idea"].replace("Approach:", "").replace("Method:", "—").strip()[:260]

  if state["feedback"] != "OK":
    entry = f"DROP ({state['feedback'][:90]}) | {summary}"
    print(f"  -> {entry}")
    log_history(entry)
    return {"history": state["history"] + [entry]}

  with open("train.py", "w", encoding="utf-8") as f:
    f.write(state["code"] + "\n")

  eval_cmd = (
    "import json, torch, prepare, train; "
    "torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None; "
    "r = prepare.evaluate(train.function); "
    "r['peak_mb'] = round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0; "
    "print('__RES__' + json.dumps(r))"
  )
  rc, out, err = run_cmd(eval_cmd, timeout=180)
  line = next((l for l in out.splitlines() if l.startswith("__RES__")), None)
  res = json.loads(line[7:]) if (rc == 0 and line) else {"valid": False, "error": (err.strip().splitlines() or ["Error"])[-1]}

  if res.get("valid") and res["time_ms"] < state["best_ms"] * 0.95:
    entry = f"KEEP {state['best_ms']}ms -> {res['time_ms']}ms (sim:{res['sim']}, peak:{res['peak_mb']}MB) | {summary}"
    print(f"  -> {entry}")
    log_history(entry)
    return {"best_code": state["code"], "best_ms": res["time_ms"], "history": state["history"] + [entry]}

  with open("train.py", "w", encoding="utf-8") as f:
    f.write(state["best_code"] + "\n")

  reason = f"{res['time_ms']}ms, sim:{res['sim']}" if res.get("valid") else res["error"][:100]
  entry = f"DROP ({reason}) | {summary}"
  print(f"  -> {entry}")
  log_history(entry)
  return {"history": state["history"] + [entry]}


def next_after_review(state: State) -> Literal["coder", "evaluator"]:
  if state["feedback"] == "OK" or state["retries"] >= 3:
    return "evaluator"
  if "accuracy" in state["feedback"] and state["feedback"] == state["prev_feedback"]:
    return "evaluator"
  return "coder"


g = StateGraph(State)
g.add_node("planner", planner)
g.add_node("coder", coder)
g.add_node("reviewer", reviewer)
g.add_node("evaluator", evaluator)
g.add_edge(START, "planner")
g.add_edge("planner", "coder")
g.add_edge("coder", "reviewer")
g.add_conditional_edges("reviewer", next_after_review, {"coder": "coder", "evaluator": "evaluator"})
g.add_conditional_edges("evaluator", lambda s: END if s["step"] >= MAX_STEPS else "planner")
app = g.compile()


if __name__ == "__main__":
  t0 = time.perf_counter()
  base_code = open("train.py", encoding="utf-8").read().strip()

  init_cmd = "import json, prepare, train; r = prepare.evaluate(train.function); r['n'] = int(prepare.OBJECT_DATA.shape[0]); print('__RES__' + json.dumps(r))"
  _, out, err = run_cmd(init_cmd, timeout=300)
  line = next((l for l in out.splitlines() if l.startswith("__RES__")), None)
  if not line:
    raise RuntimeError(f"Baseline failed:\n{err}")

  init_res = json.loads(line[7:])
  base_ms, num_points = init_res["time_ms"], init_res["n"]
  print(f"Baseline (N={num_points:,}): {base_ms}ms")

  hist = [l.strip()[2:] for l in open(HISTORY_FILE, encoding="utf-8") if l.startswith("- ")] if os.path.exists(HISTORY_FILE) else []
  final = app.invoke({
    "best_code": base_code, "best_ms": base_ms, "num_points": num_points,
    "idea": "", "code": "", "feedback": "", "prev_feedback": "",
    "retries": 0, "history": hist, "step": 0,
  })
  print(f"\nDone! Best: {final['best_ms']}ms ({time.perf_counter() - t0:.1f}s)")