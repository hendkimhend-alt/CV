import json


class ConfigError(ValueError):
    pass


def _strip_comments(value):
    # '_'로 시작하는 키는 설명용
    if isinstance(value, dict):
        return {k: _strip_comments(v) for k, v in value.items() if not k.startswith("_")}
    return value


def load_json(path):
    try:
        return _strip_comments(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise ConfigError(f"설정 파일 없음: {path}") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"설정 파일 형식 오류 {path}: {exc}") from None


def check_keys(section, expected, where):
    # 모르는 옵션이 그냥 무시되면 안 되니까 키가 정확히 같은지 본다
    unknown = sorted(set(section) - set(expected))
    missing = sorted(set(expected) - set(section))
    if unknown:
        raise ConfigError(f"{where}: 지원하지 않는 항목 {unknown}")
    if missing:
        raise ConfigError(f"{where}: 빠진 항목 {missing}")
