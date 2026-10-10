"""전처리 설정 파일(configs/preprocessing.json)을 읽는다."""
import hashlib
import json
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "preprocessing.json"
SECTIONS = ("roi", "validation", "manual_correction", "analysis_mask", "quality", "gamma", "gaussian", "output")


class ConfigError(ValueError):
    pass


def strip_comments(value):
    # "_"로 시작하는 키는 설명용
    if isinstance(value, dict):
        return {k: strip_comments(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, list):
        return [strip_comments(v) for v in value]
    return value


def check_config(cfg):
    missing = [s for s in SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"설정에 없는 항목: {missing}")

    gamma, gaussian = cfg["gamma"], cfg["gaussian"]
    if gamma["mode"] not in ("off", "fixed", "conditional"):
        raise ConfigError("gamma.mode는 off / fixed / conditional")
    if gamma["selection"] not in ("fixed", "adaptive"):
        raise ConfigError("gamma.selection은 fixed / adaptive")
    if gaussian["mode"] not in ("off", "fixed", "conditional"):
        raise ConfigError("gaussian.mode는 off / fixed / conditional")
    if gaussian["mode"] != "off" and gaussian["sigma"] <= 0:
        raise ConfigError("gaussian을 켜려면 sigma > 0")

    odd_kernels = {
        "analysis_mask.kernel_size": cfg["analysis_mask"]["kernel_size"],
        "gaussian.kernel": gaussian["kernel"],
        "roi.refine.close_kernel": cfg["roi"]["refine"]["close_kernel"],
        "roi.refine.open_kernel": cfg["roi"]["refine"]["open_kernel"],
    }
    for name, k in odd_kernels.items():
        if k > 1 and k % 2 == 0:
            raise ConfigError(f"{name}는 홀수여야 함 (현재 {k})")
    return cfg


def load_config(path=None, overrides=None):
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"설정 파일 없음: {path}") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"설정 파일 JSON 오류 {path}: {exc}") from None
    cfg = strip_comments(cfg)
    for item in overrides or []:
        apply_override(cfg, item)
    return check_config(cfg)


def apply_override(cfg, item):
    """'gamma.value=0.9' 형식의 덮어쓰기 하나를 적용한다."""
    if "=" not in item:
        raise ConfigError(f"덮어쓰기 형식은 키.경로=값 (현재 {item!r})")
    key, text = item.split("=", 1)
    parts = key.strip().split(".")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text

    node = cfg
    for part in parts[:-1]:
        if part not in node:
            raise ConfigError(f"없는 설정 키: {key}")
        node = node[part]
    if parts[-1] not in node:
        raise ConfigError(f"없는 설정 키: {key}")
    node[parts[-1]] = value


def config_hash(cfg):
    text = json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
