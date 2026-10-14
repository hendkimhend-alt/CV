"""설정 파일(json) 읽기와 키 검사."""
import json


class ConfigError(ValueError):
    """설정 파일이 없거나 형식, 키가 틀렸을 때 내는 오류."""
    pass


def _strip_comments(value):
    """설정에서 설명용 키를 지운다 (안쪽 dict까지)."""
    # '_'로 시작하는 키는 설명용
    if isinstance(value, dict):
        return {k: _strip_comments(v) for k, v in value.items() if not k.startswith("_")}
    return value


def load_json(path):
    """json 설정 파일을 읽어 설명용 키를 뺀 dict로 돌려준다. 없거나 형식이 틀리면 ConfigError."""
    try:
        return _strip_comments(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise ConfigError(f"설정 파일 없음: {path}") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"설정 파일 형식 오류 {path}: {exc}") from None


def check_keys(section, expected, where):
    """section의 키가 expected와 정확히 같은지 본다. 모르는 키나 빠진 키가 있으면 ConfigError."""
    # 모르는 옵션이 그냥 무시되면 안 되니까 키가 정확히 같은지 본다
    unknown = sorted(set(section) - set(expected))
    missing = sorted(set(expected) - set(section))
    if unknown:
        raise ConfigError(f"{where}: 지원하지 않는 항목 {unknown}")
    if missing:
        raise ConfigError(f"{where}: 빠진 항목 {missing}")
