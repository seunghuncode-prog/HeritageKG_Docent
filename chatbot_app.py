"""
국립중앙박물관 AI 도슨트 - Streamlit UI
실행: streamlit run chatbot_app.py
"""

import uuid
import streamlit as st
import sys
import os
import requests
sys.path.insert(0, os.path.dirname(__file__))
from prompts.style_prompt import get_style_prompt, Persona, PERSONA_GUIDES

# ── API 설정 ──────────────────────────────────────────────────────────────────
GRAPH_RAG_URL = "http://localhost:8000"

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="국립중앙박물관 AI 도슨트",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

* { font-family: 'Noto Sans KR', sans-serif; }

/* 전체 배경 */
.stApp { background-color: #f8f5f0; }

/* 사이드바 */
[data-testid="stSidebar"] {
    background-color: #1a2744;
}
[data-testid="stSidebar"] * { color: #e8dcc8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #c9a84c !important; }

/* 헤더 숨기기 */
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* 채팅 입력창 */
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
    border: 2px solid #c9a84c !important;
    background-color: #ffffff !important;
    font-size: 15px !important;
    padding: 12px 20px !important;
}
[data-testid="stChatInput"] textarea:focus {
    box-shadow: 0 0 0 3px rgba(201,168,76,0.2) !important;
}

/* 어시스턴트 메시지 */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #ffffff;
    border-radius: 16px;
    padding: 4px 12px;
    border-left: 4px solid #c9a84c;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 12px;
}

/* 유저 메시지 */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #eef2fb;
    border-radius: 16px;
    padding: 4px 12px;
    margin-bottom: 12px;
}

/* 타이틀 영역 */
.main-title {
    text-align: center;
    padding: 20px 0 8px 0;
    color: #1a2744;
}
.main-title h1 { font-size: 26px; font-weight: 700; margin-bottom: 4px; }
.main-title p  { font-size: 14px; color: #7a6a50; margin: 0; }

.divider {
    border: none;
    border-top: 1px solid #e0d8c8;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)


# ── 세션 초기화 ───────────────────────────────────────────────────────────────
if "messages"     not in st.session_state:
    st.session_state.messages     = []
if "persona"      not in st.session_state:
    st.session_state.persona      = Persona.BEGINNER
if "session_id"   not in st.session_state:
    st.session_state.session_id   = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏛️ AI 도슨트")
    st.markdown("**국립중앙박물관** 유물 안내 서비스")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── 페르소나 선택 ──────────────────────────────────────────────────────
    st.markdown("### 👤 관람객 유형")
    persona_labels = {
        Persona.CHILD:    "🧒 어린이",
        Persona.BEGINNER: "🙂 초보",
        Persona.EXPERT:   "🎓 역사 전문가",
    }
    persona_descriptions = {
        Persona.CHILD:    "쉬운 말로 재미있게!",
        Persona.BEGINNER: "친절하게 기본 설명",
        Persona.EXPERT:   "깊이 있는 학술 정보",
    }

    selected_label = st.radio(
        label="유형 선택",
        options=list(persona_labels.values()),
        index=list(persona_labels.keys()).index(st.session_state.persona),
        label_visibility="collapsed",
    )
    selected_persona = next(p for p, l in persona_labels.items() if l == selected_label)
    if selected_persona != st.session_state.persona:
        st.session_state.persona = selected_persona

    st.caption(persona_descriptions[st.session_state.persona])
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("### 📊 DB 현황")
    st.markdown("""
    | 항목 | 수량 |
    |------|------|
    | 유물 | 2,560개 |
    | 시대 | 72종 |
    | 재질 | 60종 |
    | 분류 | 373종 |
    """)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    if st.button("🗑️ 대화 초기화", key="clear"):
        st.session_state.messages     = []
        st.session_state.chat_history = []
        st.session_state.session_id   = str(uuid.uuid4())
        st.rerun()


# ── 메인 영역 ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-title">
    <h1>🏛️ 국립중앙박물관 AI 도슨트</h1>
    <p>유물에 대해 무엇이든 물어보세요</p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)


# 메시지 없을 때 환영 화면
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #9a8a70;">
        <div style="font-size: 56px; margin-bottom: 16px;">🏺</div>
        <div style="font-size: 18px; font-weight: 500; color: #3a3020; margin-bottom: 8px;">
            안녕하세요! 국립중앙박물관 AI 도슨트입니다.
        </div>
        <div style="font-size: 14px; line-height: 1.8;">
            유물의 역사, 시대적 배경, 제작 방식 등<br>
            궁금한 것을 자유롭게 질문해보세요.<br><br>
        </div>
    </div>
    """, unsafe_allow_html=True)


# 채팅 히스토리 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🏛️" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])


# ── API 호출 ──────────────────────────────────────────────────────────────────
def get_response(message: str, persona: Persona) -> str:
    """Graph RAG API(port 8000) 호출 → 답변 반환"""
    try:
        resp = requests.post(
            f"{GRAPH_RAG_URL}/chat",
            json={
                "question":   message,
                "persona":    persona.value,
                "session_id": st.session_state.session_id,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("answer", "답변을 생성하지 못했습니다.")
    except requests.exceptions.ConnectionError:
        return "⚠️ API 서버에 연결할 수 없습니다.\n\n`python graph_rag_api.py`를 먼저 실행해주세요."
    except Exception as e:
        return f"⚠️ 오류 발생: {e}"


# ── 입력 처리 ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("유물에 대해 무엇이든 물어보세요...")

if user_input:
    # 유저 메시지
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # 어시스턴트 응답
    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("유물 정보를 검색하는 중..."):
            answer = get_response(user_input, st.session_state.persona)
        st.markdown(answer)

    # 히스토리 저장
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append({"role": "user",      "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
