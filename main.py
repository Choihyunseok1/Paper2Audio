import os
import arxiv
import openai
from notion_client import Client
import datetime

# 깃허브 금고에서 열쇠 가져오기
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["DATABASE_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

notion = Client(auth=NOTION_TOKEN)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def run_bot():
    search = arxiv.Search(query="cat:cs.CV", max_results=3, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    # 내 깃허브 사용자 이름과 저장소 이름을 여기에 맞게 수정하세요!
    GITHUB_USER = "본인의_깃허브_ID"
    GITHUB_REPO = "daily-ai-podcast"

    for result in search.results():
        # 파일명을 논문 ID로 고정 (예: 2401.1234.mp3)
        paper_id = result.entry_id.split('/')[-1]
        file_path = f"audio/{paper_id}.mp3"
        
        # 폴더가 없으면 생성
        os.makedirs("audio", exist_ok=True)

        # (생략: GPT 요약 및 TTS 생성 로직은 이전과 동일)
        # audio_response.stream_to_file(file_path)

        # 팟캐스트 파일의 "진짜 인터넷 주소" 만들기
        audio_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/{file_path}"

        # 노션 등록 (이제 '오디오' 칸에 진짜 주소가 들어갑니다)
        notion.pages.create(
            parent={"database_id": DATABASE_ID},
            properties={
                "이름": {"title": [{"text": {"content": result.title}}]},
                "날짜": {"date": {"start": datetime.date.today().isoformat()}},
                "카테고리": {"select": {"name": "CV"}},
                "요약": {"rich_text": [{"text": {"content": summary_text}}]},
                "오디오": {"url": audio_url} # <--- 핵심: 바로 재생 가능한 링크!
            }
        )
