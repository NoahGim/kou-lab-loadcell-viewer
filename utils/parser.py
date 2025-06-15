import re

def parse_loadcell(line: str):
    """
    [LoadCell_input] Cell 1: -57 | Cell 4: 100 ...
    형식의 문자열에서 (셀 인덱스, 값) 튜플의 리스트를 반환
    (예: [(0, -57), (3, 100)])
    """
    if not line.startswith('[LoadCell_input]'):
        return None
    # 'Cell <number>: <value>' 패턴을 모두 찾아 숫자와 값을 캡처 (소수점 포함)
    pattern = r'Cell\s*(\d+):\s*([\-\d\.]+)'
    matches = re.findall(pattern, line)
    if matches:
        # 셀 번호는 1-based이므로 0-based 인덱스로, 값은 float으로 변환
        return [(int(num) - 1, float(val)) for num, val in matches]
    return None

def parse_motor(line: str):
    """
    [Motor] RPM: 0.00 | PWM: 255
    형식의 문자열에서 RPM, PWM 값을 튜플로 반환
    """
    if not line.startswith('[Motor]'):
        return None
    pattern = r'RPM: ([\d\.]+) \| PWM: ([\d]+)'
    match = re.search(pattern, line)
    if match:
        return float(match.group(1)), int(match.group(2))
    return None 