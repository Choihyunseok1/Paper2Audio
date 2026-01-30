import os
import re
import io
import math
import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import arxiv
import openai
import requests
from notion_client import Client as NotionClient
from zoneinfo import ZoneInfo
from pydub import AudioSegment


# =========================
# ENV
# =========================
NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "").strip()
DATABASE_ID = os.environ.get("DATABASE_ID", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()

GITHUB_USER = os.environ.get("GITHUB_USER", "Choihyunseok1")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "CV_Papers_Podtcast_bot")

if not NOTION_TOKEN or not DATABASE_ID or not OPENAI_API_KEY:
    raise RuntimeError("Missing env: NOTION_TOKEN / DATABASE_ID / OPENAI_API_KEY")

notion = NotionClient(auth=NOTION_TOKEN)
client = openai.OpenAI(api_key=OPENAI_API_KEY)


# =========================
# CONFIG
# =========================
ARXIV_CATEGORY_QUERY = "cat:cs.CV"
ARXIV_MAX_RESULTS = 800  # 넉넉히 (필요하면 더 올려도 됨)

TOPK_SAVE = 10
PRESELECT_K = 60  # Semantic Scholar 호출 수를 줄이기 위한 1차 후보 수

BATCH_SIZE_FULL = 2

MAX_OUT_TOKENS_SUMMARY = 4000
MAX_OUT_TOKENS_FULL_PER_BATCH = 2800

TTS_MODEL = "tts-1-hd"
TTS_VOICE = "onyx"
TTS_SPEED = 1.25

TTS_CHUNK_CHARS = 2000
TTS_CHUNK_OVERLAP = 0

SEOUL_TZ = ZoneInfo("Asia/Seoul")
ET_TZ = ZoneInfo("America/New_York")

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_TIMEOUT = 10

# =========================
# SCORING (가벼운 버전)
# - Author score 비중 크게
# - PDF 파싱/affiliation 등 무거운 기준 제거
# =========================
POSITIVE_KEYWORDS = [
    "transformer", "diffusion", "attention", "multimodal", "vision-language",
    "end-to-end", "benchmark", "dataset", "large-scale", "foundation model",
    "distillation", "self-supervised", "contrastive", "retrieval", "tracking",
    "segmentation", "detection", "3d", "point cloud", "neural rendering",
    "video", "temporal", "efficient", "real-time"
]
NEGATIVE_HYPE = ["revolutionary", "breakthrough", "sota", "state-of-the-art"]
CODE_HINTS = ["github.com", "gitlab.com", "bitbucket.org", "project page", "code"]


@dataclass
class ScoredPaper:
    paper: Any
    arxiv_id: str
    score: float
    score_detail: Dict[str, float]


# =========================
# TIME WINDOW: arXiv announce 기준
# (ET 14:00 cutoff / ET 20:00 announce)
# - Tue/Wed/Thu: (prev day 14:00) ~ (today 14:00)
# - Mon: (Fri 14:00) ~ (Mon 14:00)
# - Sun: (Thu 14:00) ~ (Fri 14:00)  # arXiv가 주말에 Sun 20:00에 배치 발표
# =========================
def is_announce_day_et(d: datetime.date) -> bool:
    # Mon(0), Tue(1), Wed(2), Thu(3), Sun(6)
    return d.weekday() in (0, 1, 2, 3, 6)


def latest_announce_date_et(now_et: datetime.datetime) -> datetime.date:
    d = now_et.date()
    # 오늘이 announce day가 아니거나, announce time(20:00) 이전이면 이전 announce day로 후퇴
    if (not is_announce_day_et(d)) or (now_et.time() < datetime.time(20, 0)):
        d = d - datetime.timedelta(days=1)
        while not is_announce_day_et(d):
            d = d - datetime.timedelta(days=1)
        return d
    return d


def compute_announce_window_et(now_et: datetime.datetime) -> Tuple[datetime.datetime, datetime.datetime, datetime.date]:
    ad = latest_announce_date_et(now_et)

    # end cutoff
    if ad.weekday() == 6:
        # Sunday announce -> window end = Friday 14:00 ET (ad - 2 days)
        end_date = ad - datetime.timedelta(days=2)  # Friday
        end_et = datetime.datetime(end_date.year, end_date.month, end_date.day, 14, 0, tzinfo=ET_TZ)
        start_et = end_et - datetime.timedelta(days=1)  # Thursday 14:00
        return start_et, end_et, ad

    if ad.weekday() == 0:
        # Monday announce -> window end = Monday 14:00, start = Friday 14:00
        end_et = datetime.datetime(ad.year, ad.month, ad.day, 14, 0, tzinfo=ET_TZ)
        start_et = end_et - datetime.timedelta(days=3)
        return start_et, end_et, ad

    # Tue/Wed/Thu announce
    end_et = datetime.datetime(ad.year, ad.month, ad.day, 14, 0, tzinfo=ET_TZ)
    start_et = end_et - datetime.timedelta(days=1)
    return start_et, end_et, ad


# =========================
# UTIL
# =========================
def split_notion_text(text: str, max_len: int = 1900) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    return [t[i:i + max_len] for i in range(0, len(t), max_len)]


def chunk_text_by_chars(text: str, chunk_chars: int = 2000, overlap: int = 0) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    chunks = []
    i = 0
    n = len(t)
    step = max(1, chunk_chars - overlap)
    while i < n:
        chunk = t[i:i + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def sanitize_title_for_tts(title: str) -> str:
    if not title:
        return ""
    return re.sub(r"[:\-+/]", ",", title)


def extract_arxiv_id(entry_id: str) -> str:
    # example entry_id: http://arxiv.org/abs/2401.12345v2
    if not entry_id:
        return ""
    m = re.search(r"arxiv\.org/abs/([^v]+)(?:v\d+)?", entry_id)
    if m:
        return m.group(1).strip()
    # fallback: last token
    return entry_id.rsplit("/", 1)[-1].replace("v1", "").replace("v2", "").strip()


def safe_lower(s: str) -> str:
    return (s or "").lower()


# =========================
# ARXIV FETCH
# =========================
def fetch_arxiv_candidates_in_window(start_et: datetime.datetime, end_et: datetime.datetime) -> List[Any]:
    aclient = arxiv.Client(page_size=200, delay_seconds=0.0, num_retries=3)

    search = arxiv.Search(
        query=ARXIV_CATEGORY_QUERY,
        max_results=ARXIV_MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    candidates = []
    for p in aclient.results(search):
        # arxiv library gives timezone-aware datetime in UTC generally
        pub_et = p.published.astimezone(ET_TZ)

        # SubmittedDate desc라서 pub_et이 start_et보다 작아지면 이후는 더 과거 -> break
        if pub_et < start_et:
            break

        if start_et <= pub_et < end_et:
            candidates.append(p)

    return candidates


# =========================
# STAGE 0: LIGHT PRE-SCORE
# =========================
def prescore_text(p: Any) -> float:
    title = safe_lower(p.title)
    abst = safe_lower(p.summary)
    text = title + " " + abst

    kw_hits = sum(1 for k in POSITIVE_KEYWORDS if k in text)
    hype_hits = sum(1 for k in NEGATIVE_HYPE if k in text)
    code_hit = any(h in text for h in CODE_HINTS)

    score = 0.0
    score += min(kw_hits * 2.0, 18.0)       # 0~18
    score += 6.0 if code_hit else 0.0       # +6
    score -= min(hype_hits * 2.0, 6.0)      # -0~-6
    return score


# =========================
# SEMANTIC SCHOLAR
# =========================
def s2_headers() -> Dict[str, str]:
    h = {"User-Agent": "cv-papers-podcast-bot/1.0"}
    if SEMANTIC_SCHOLAR_API_KEY:
        h["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    return h


def fetch_s2_paper_by_arxiv(arxiv_id: str) -> Optional[Dict[str, Any]]:
    if not arxiv_id:
        return None
    url = f"{S2_BASE}/paper/ARXIV:{arxiv_id}"
    fields = "title,authors.name,authors.hIndex,authors.paperCount,externalIds,url"
    try:
        r = requests.get(url, headers=s2_headers(), params={"fields": fields}, timeout=S2_TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def compute_author_score(s2: Optional[Dict[str, Any]]) -> float:
    # Author score: 0~70 (비중 크게)
    if not s2 or "authors" not in s2:
        return 10.0  # 정보 없으면 바닥 너무 낮게 깔지 않되, 낮은 점수

    authors = s2.get("authors") or []
    if not authors:
        return 10.0

    h_list = []
    pc_list = []
    for a in authors:
        h = a.get("hIndex")
        pc = a.get("paperCount")
        if isinstance(h, int):
            h_list.append(h)
        if isinstance(pc, int):
            pc_list.append(pc)

    if not h_list:
        h_list = [0]

    # top-2 평균 (강한 저자 1~2명 반영)
    h_list_sorted = sorted(h_list, reverse=True)
    top2 = h_list_sorted[:2]
    h_avg = sum(top2) / max(1, len(top2))

    # piecewise (대충 연구자 필터링 감각)
    if h_avg >= 40:
        base = 62.0
    elif h_avg >= 25:
        base = 52.0
    elif h_avg >= 15:
        base = 40.0
    elif h_avg >= 8:
        base = 28.0
    else:
        base = 18.0

    # paperCount 보정 (너무 과하지 않게)
    pc_max = max(pc_list) if pc_list else 0
    if pc_max >= 200:
        base += 6.0
    elif pc_max >= 100:
        base += 4.0
    elif pc_max >= 50:
        base += 2.0

    return max(0.0, min(70.0, base))


def compute_content_score(p: Any) -> float:
    # Content score: 0~25 (가벼운 텍스트 신호)
    title = safe_lower(p.title)
    abst = safe_lower(p.summary)
    text = title + " " + abst

    kw_hits = sum(1 for k in POSITIVE_KEYWORDS if k in text)
    code_hit = any(h in text for h in CODE_HINTS)

    score = 0.0
    score += min(kw_hits * 2.0, 18.0)   # 0~18
    score += 7.0 if code_hit else 0.0   # +7
    return max(0.0, min(25.0, score))


def compute_penalty(p: Any) -> float:
    # penalty: -10~0
    title = safe_lower(p.title)
    abst = safe_lower(p.summary)
    text = title + " " + abst

    hype_hits = sum(1 for k in NEGATIVE_HYPE if k in text)
    if hype_hits == 0:
        return 0.0

    # 과장 표현이 있으면 약하게 감점
    return -min(10.0, hype_hits * 3.0)


def score_paper(p: Any) -> ScoredPaper:
    arxiv_id = extract_arxiv_id(getattr(p, "entry_id", ""))
    s2 = fetch_s2_paper_by_arxiv(arxiv_id)

    author = compute_author_score(s2)      # 0~70
    content = compute_content_score(p)     # 0~25
    penalty = compute_penalty(p)           # -10~0

    total = author + content + penalty     # 0~95 정도 범위
    # 0~100 스케일로 보기 좋게 약간 확장
    total = max(0.0, min(100.0, total * (100.0 / 95.0)))

    return ScoredPaper(
        paper=p,
        arxiv_id=arxiv_id,
        score=total,
        score_detail={"author": author, "content": content, "penalty": penalty}
    )


# =========================
# GPT PROMPTS (Top10만)
# =========================
def build_papers_info(papers: List[Any]) -> str:
    out = []
    for i, p in enumerate(papers, start=1):
        out.append(f"논문 {i} 제목: {p.title}\n초록: {p.summary}\n")
    return "\n".join(out).strip()


def prompt_summary_and_3min(top_papers: List[Any], window_label: str) -> str:
    papers_info = build_papers_info(top_papers)
    return f"""
아래는 arXiv cs.CV에서 "{window_label}"에 해당하는 배치에서, 임의 스코어링으로 상위 {len(top_papers)}편만 선별한 논문입니다.
(중요: 아래에 포함된 논문만 요약/오디오 대본을 작성하십시오. 나머지 논문은 절대 다루지 마십시오.)

{papers_info}

위 논문들을 바탕으로 다음 두 가지를 작성해 주세요.

1. [요약]
- 노션 기록용 핵심 요약.
- 각 논문별로 제목을 언급하고, '-함', '-임' 형태의 짧은 요약체로 3줄씩 작성할 것
- 한 줄이 끝나면 반드시 엔터로 구분해서 보기 편하게 만들 것
- 각 논문 요약 시작시 '1. (논문제목)' 식으로 앞에 번호만 붙여 진행할 것
- 논문들 사이는 줄바꿈으로 구분할 것.

2. [3분대본]
- "시간이 없으신 분들을 위한 3분 핵심 요약입니다"로 시작할 것.
- 위에 제공된 상위 {len(top_papers)}편 논문을 빠짐없이 포함할 것.
- 각 논문 제목을 말한 뒤, 논문 당 길이를 자동으로 조절하여 전체가 약 3분(±15초)이 되도록 구성할 것.
- 논문의 공식 제목은 반드시 영문으로 표기하되, 제목의 특수 기호(:, -, +, / 등)는 쉼표(,)로 바꿀 것.
- CNN, ViT, GAN, SOTA 등 약어는 영문 그대로 사용할 것.
- 전문 기술 용어는 번역하지 말고 반드시 영어 원어 그대로 사용할 것.
- 쉼표(,)를 충분히 사용해 호흡 지점을 표시할 것.
- 전체 브리핑은 공적인 라디오 방송 톤의 존댓말로 작성할 것.
- 반말, 구어체 축약, 친근한 대화체(예: ~해요, ~했죠)는 사용하지 말 것.
- 연구 비서가 공식적으로 설명하는 말투를 유지할 것.

[마무리 규칙]
- 모든 논문 설명이 끝난 뒤, 반드시 아래 톤의 아웃트로 멘트를 한 문단으로 추가할 것.
- 감사 인사나 일상적인 인삿말은 사용하지 말 것.
- 더 자세한 내용이 전체 브리핑에 있다는 점을 자연스럽게 안내할 것.

아웃트로 예시 톤:
"보다 자세한 내용은 전체 브리핑에서 이어서 다룹니다.
지금까지 오늘의 컴퓨터 비전 논문 3분 핵심 요약이었습니다."

출력 형식:
[요약]
(내용)

[3분대본]
(내용)
""".strip()


def prompt_full_body_for_batch(batch_papers: List[Any], batch_index: int, total_batches: int) -> str:
    papers_info = build_papers_info(batch_papers)

    return f"""
아래는 상위 선별 논문 배치 {batch_index}/{total_batches}입니다.
(중요: 아래에 포함된 논문만 본문 스크립트를 작성하십시오.)

{papers_info}

중요:
- 방송의 도입부와 맺음말을 쓰지 않습니다.
- "첫 번째 논문", "이번 배치", "안녕하세요", "오늘은" 같은 진행 멘트와 순서 멘트를 절대 쓰지 마세요.
- 오직 각 논문 설명 본문만 출력하세요.

분량:
- 논문 1편당 약 1700~2200자 내외로 상세히 설명하세요.

언어 규칙:
- CNN, ViT, GAN, SOTA 등 약어는 영문 그대로 사용할 것.
- 전문 기술 용어는 번역하지 말고 영어 원어 그대로 사용할 것.
- 쉼표(,)로 호흡, 마침표(.)로 강조.
- 공적인 라디오 방송 톤의 존댓말.
- 반말, 구어체 축약, 친근한 대화체(예: ~해요, ~했죠) 금지.
- "A.", "B.", "첫째", "다음으로", "이어서" 등 구조/순서를 직접적으로 언급하지 말 것.

출력 형식(반드시 준수):
TITLE: <영문 제목>
BODY:
<본문>

(논문과 논문 사이는 빈 줄 2줄)
""".strip()


def call_gpt_text(system_text: str, user_text: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        max_tokens=max_tokens
    )
    return (resp.choices[0].message.content or "").strip()


# =========================
# TTS
# =========================
def synthesize_tts_to_audio(text: str, tts_chunk_chars: int = 2000, overlap: int = 0) -> AudioSegment:
    chunks = chunk_text_by_chars(text, chunk_chars=tts_chunk_chars, overlap=overlap)
    combined = AudioSegment.empty()
    for chunk in chunks:
        audio_part_response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=chunk,
            speed=TTS_SPEED
        )
        part_stream = io.BytesIO(audio_part_response.content)
        segment = AudioSegment.from_file(part_stream, format="mp3")
        combined += segment
    return combined


def parse_title_body_blocks(text: str) -> List[Tuple[str, str]]:
    t = (text or "").strip()
    if not t:
        return []

    pattern = r"TITLE:\s*(.*?)\s*BODY:\s*(.*?)(?=(?:\n\s*TITLE:)|\Z)"
    matches = re.findall(pattern, t, flags=re.DOTALL | re.IGNORECASE)

    blocks = []
    for title, body in matches:
        tt = title.strip()
        bb = body.strip()
        if tt and bb:
            blocks.append((tt, bb))
    return blocks


def assemble_radio_script(full_batches_text: List[str], total_papers: int, window_label: str) -> str:
    intro = f"안녕하세요, 아이알씨브이 랩실의 수석 연구 비서입니다. 지금부터 arXiv cs.CV에서 {window_label} 배치 기준으로 선별된 상위 {total_papers}편을 브리핑 드리겠습니다."
    outro = "이상으로 오늘의 전체 브리핑을 마칩니다."

    all_blocks = []
    for batch_text in full_batches_text:
        all_blocks.extend(parse_title_body_blocks(batch_text))

    script_parts = [intro, ""]
    for i, (title, body) in enumerate(all_blocks, start=1):
        title_tts = sanitize_title_for_tts(title)
        script_parts.append(f"{i}번째 논문입니다.")
        script_parts.append(f"논문 제목은 {title_tts} 입니다.")
        script_parts.append(body)
        script_parts.append("")

    if len(all_blocks) < total_papers:
        script_parts.append("일부 원고 생성이 누락되어, 생성된 부분까지만 읽었습니다.")
        script_parts.append("")

    script_parts.append(outro)
    return "\n".join(script_parts).strip()


# =========================
# NOTION
# =========================
def notion_rich_link(text: str, url: str) -> Dict[str, Any]:
    return {
        "type": "text",
        "text": {"content": text, "link": {"url": url}}
    }


def run_bot():
    now_kst = datetime.datetime.now(SEOUL_TZ)
    now_et = now_kst.astimezone(ET_TZ)

    start_et, end_et, announce_date = compute_announce_window_et(now_et)

    window_label = f"{announce_date.isoformat()} announce, submissions window {start_et.strftime('%m/%d %H:%M')}~{end_et.strftime('%m/%d %H:%M')} ET"

    candidates = fetch_arxiv_candidates_in_window(start_et, end_et)
    if not candidates:
        print("해당 announce window에서 새로 잡힌 논문이 없습니다.")
        return

    # 1차: 텍스트 예비 점수로 PRESELECT_K로 압축
    prescored = [(p, prescore_text(p)) for p in candidates]
    prescored.sort(key=lambda x: x[1], reverse=True)
    preselected = [p for (p, _) in prescored[:min(PRESELECT_K, len(prescored))]]

    # 2차: Semantic Scholar + Author 비중 큰 최종 점수
    scored: List[ScoredPaper] = []
    for p in preselected:
        scored.append(score_paper(p))

    scored.sort(key=lambda sp: sp.score, reverse=True)
    top = scored[:min(TOPK_SAVE, len(scored))]
    top_papers = [sp.paper for sp in top]

    if not top_papers:
        print("TopK 선별 결과가 비어 있습니다.")
        return

    # GPT 생성 (Top10만)
    system_summary = "너는 IRCV 랩실의 수석 연구 비서다. 한국어로 요약과 3분 대본을 공적인 존댓말 톤으로 작성하라."
    system_full = "너는 IRCV 랩실의 수석 연구 비서다. 한국어로 논문 본문 스크립트만 공적인 존댓말 톤으로 작성하라."

    user_summary = prompt_summary_and_3min(top_papers, window_label)
    summary_out = call_gpt_text(system_summary, user_summary, MAX_OUT_TOKENS_SUMMARY)

    if "[3분대본]" in summary_out:
        summary_text = summary_out.split("[3분대본]")[0].replace("[요약]", "").strip()
        audio_script_3min = summary_out.split("[3분대본]")[1].strip()
    else:
        summary_text = summary_out.replace("[요약]", "").strip()
        audio_script_3min = ""

    # Full body batches
    paper_batches = [top_papers[i:i + BATCH_SIZE_FULL] for i in range(0, len(top_papers), BATCH_SIZE_FULL)]
    total_batches = len(paper_batches)

    full_batches_text = []
    for idx, batch in enumerate(paper_batches, start=1):
        user_full = prompt_full_body_for_batch(batch, idx, total_batches)
        batch_text = call_gpt_text(system_full, user_full, MAX_OUT_TOKENS_FULL_PER_BATCH)
        full_batches_text.append(batch_text)

    audio_script_full = assemble_radio_script(full_batches_text, total_papers=len(top_papers), window_label=window_label)

    # TTS
    base_path = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    today_date = now_kst.strftime("%Y%m%d")
    file_name_full = f"CV_Daily_Briefing_{today_date}.mp3"
    file_name_3min = f"3Min_Summary_{today_date}.mp3"

    full_file_path = os.path.join(audio_dir, file_name_full)
    full_file_path_3min = os.path.join(audio_dir, file_name_3min)

    combined_audio = synthesize_tts_to_audio(
        audio_script_full,
        tts_chunk_chars=TTS_CHUNK_CHARS,
        overlap=TTS_CHUNK_OVERLAP
    )
    combined_audio.export(full_file_path, format="mp3")

    if audio_script_3min.strip():
        audio_3min = synthesize_tts_to_audio(
            audio_script_3min,
            tts_chunk_chars=TTS_CHUNK_CHARS,
            overlap=TTS_CHUNK_OVERLAP
        )
        audio_3min.export(full_file_path_3min, format="mp3")
    else:
        open(full_file_path_3min, "wb").close()

    # URLs
    audio_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/audio/{file_name_full}"
    audio_url_3min = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/audio/{file_name_3min}"

    page_title = f"{now_kst.strftime('%Y-%m-%d')} 모닝 브리핑 Top{len(top_papers)}"

    # Notion children
    notion_children = [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": f"announce window: {window_label}"}}]}
        },
        {"object": "block", "type": "divider", "divider": {}},
        {"object": "block", "type": "heading_2",
         "heading_2": {"rich_text": [{"type": "text", "text": {"content": "논문 핵심 요약"}}]}},
    ]

    for part in split_notion_text(summary_text, max_len=1900):
        notion_children.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": part}}]}
        })

    notion_children += [
        {"object": "block", "type": "divider", "divider": {}},
        {"object": "block", "type": "heading_2",
         "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Top 논문 원문 링크"}}]}},
    ]

    # Top10만 링크 + 점수도 같이 남김
    for rank, sp in enumerate(top, start=1):
        p = sp.paper
        line = f"{rank}. {p.title}  (score={sp.score:.1f}, author={sp.score_detail['author']:.1f}, content={sp.score_detail['content']:.1f})"
        notion_children.append({
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [
                    {"type": "text", "text": {"content": line + " "}},
                    {"type": "text", "text": {"content": "PDF", "link": {"url": p.pdf_url}},
                     "annotations": {"bold": True, "color": "blue"}}
                ]
            }
        })

    notion.pages.create(
        parent={"database_id": DATABASE_ID},
        properties={
            "요약 & 논문링크": {"title": [{"text": {"content": page_title}}]},
            "날짜": {"date": {"start": now_kst.date().isoformat()}},
            "전체 브리핑": {"rich_text": [notion_rich_link("▶ 전체 브리핑 다운", audio_url)]},
            "3분 요약": {"rich_text": [notion_rich_link("▶ 3분 요약 다운", audio_url_3min)]},
        },
        children=notion_children
    )

    print(f"완료: candidates={len(candidates)}, preselected={len(preselected)}, saved_top={len(top_papers)}")


if __name__ == "__main__":
    run_bot()
