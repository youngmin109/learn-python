import sys
import os

# 비표준 경로 추가 : 개인 환경에 맞게 수정
module_path = '/path/to/your/external_module'
if module_path not in sys.path:
    sys.path.append(module_path)
    
# 모듈 임포트
import external_module  

def main():
    greeting = external_module.greet("Python User")
    print(greeting)
    
if __name__ == '__main__':
    main()