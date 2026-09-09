import os
from dotenv import load_dotenv
from dotenv import find_dotenv
from google import genai
from google.genai import types
from prompts.system_prompt import SYSTEM_PROMPT
from prompts.safety_prompt import SAFETY_PROMPT
from prompts.memory_prompt import MEMORY_PROMPT
from prompts.graph_prompt import GRAPH_PROMPT
from prompts.style_prompt import STYLE_PROMPT
# ==================================================
# ENV
# ==================================================
load_dotenv(find_dotenv())
client = genai.Client()
# ==================================================
# LLM
# ==================================================
class LLMGenerator:
    def generate(
        self,
        question,
        context,
        chat_history
    ):
        # ==================================================
        # 대화 기록 정리
        # ==================================================
        history_text = ""
        for chat in chat_history:
            history_text += (
                f"{chat['role']}: "
                f"{chat['content']}\n"
            )
        # ==================================================
        # SYSTEM INSTRUCTION
        # ==================================================
        full_system_instruction = f"""
{SYSTEM_PROMPT}
{SAFETY_PROMPT}
{MEMORY_PROMPT}
{GRAPH_PROMPT}
{STYLE_PROMPT}
"""
        #당신은 문화유산 도슨트 입니다. 질문에 응답하제요.
        
        # ==================================================
        # USER CONTENT
        # ==================================================
        user_content = f"""
아래 검색 정보를 기반으로만 답변하세요.
[대화 기록]
{history_text}
[검색 정보]
{context}
[현재 질문]
{question}
""".strip()
        # ==================================================
        # GEMINI
        # ==================================================
        try:
            response = (
                client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=
                        full_system_instruction
                    )
                )
            )
            return response.text
        except Exception as e:
            print(
                f"❌ 에러 발생: {e}"
            )
            return (
                "답변 생성 중 오류가 발생했습니다."
            )