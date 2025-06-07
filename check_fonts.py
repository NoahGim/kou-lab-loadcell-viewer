import matplotlib.font_manager as fm

print("--- 시스템에 설치된 한글 지원 가능 폰트 목록 ---")

# 시스템 폰트 중에서 'ttf'와 'ttc' 확장자를 가진 모든 폰트 경로를 찾습니다.
font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
font_paths.extend(fm.findSystemFonts(fontpaths=None, fontext='ttc'))

for font_path in font_paths:
    try:
        # 폰트 경로로부터 Matplotlib가 인식하는 공식 이름(family name)을 가져옵니다.
        font_name = fm.FontProperties(fname=font_path).get_name()
        
        # 이름에 한글 폰트를 식별할 수 있는 키워드가 포함되어 있는지 확인합니다.
        # (대소문자 구분 없이 'gothic', 'nanum', 'malgun', 'apple' 등을 찾습니다.)
        if any(keyword in font_name.lower() for keyword in ['gothic', 'nanum', 'malgun', 'apple', 'dotum', 'gulim']):
            print(f"✅ 발견된 폰트 이름: {font_name}")
    except Exception:
        # 간혹 손상되거나 읽을 수 없는 폰트 파일이 있을 경우, 오류를 무시하고 계속 진행합니다.
        continue

print("\n--- 진단 완료 ---")
print("위 목록에서 'Apple SD Gothic Neo', 'AppleGothic', 'NanumGothic' 등의 이름을 확인하세요.")