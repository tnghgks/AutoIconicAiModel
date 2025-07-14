# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import infer  # 모델의 추론 함수
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # 허용할 origin 목록
    allow_credentials=False,           # 쿠키/인증정보 허용 여부
    allow_methods=["*"],              # 허용할 HTTP 메서드
    allow_headers=["*"],              # 허용할 헤더
)
class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_svg(request: PromptRequest):
    try:
        svg_d = infer(request.prompt)
        svg_xml = (
            '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" '
            'stroke="black" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="{svg_d}"/></svg>'
        )
        return {"svg": svg_xml}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
