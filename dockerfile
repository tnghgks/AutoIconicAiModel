# 1. 기본 이미지 (Python + Linux 기반)
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. requirements 복사 및 설치
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 4. 전체 코드 복사
COPY . .

# 5. 포트 노출 (FastAPI 기본: 8000)
EXPOSE 8000

# 6. 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
