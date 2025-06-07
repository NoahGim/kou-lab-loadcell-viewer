import re

def parse_loadcell(line: str):
    """
    [LoadCell] Cell 1: -57 | Cell 2: 7
    형식의 문자열에서 2개 로드셀 값을 리스트로 반환
    """
    if not line.startswith('[LoadCell]'):
        return None
    # 공백과 구분자(|)에 유연한 정규식
    pattern = r'Cell 1:\s*([\-\d]+)\s*\|\s*Cell 2:\s*([\-\d]+)'
    match = re.search(pattern, line)
    if match:
        return [int(match.group(1)), int(match.group(2))]
    return None

def parse_motor(line: str):
    """
    [Motor] RPM:1203, PWM: 122
    형식의 문자열에서 RPM, PWM 값을 튜플로 반환
    """
    if not line.startswith('[Motor]'):
        return None
    pattern = r'RPM:([\d]+), PWM: ([\d]+)'
    match = re.search(pattern, line)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None 