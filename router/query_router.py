class QueryRouter:

    def route(self, query):

        # 추천형 질문
        if (
            "비슷" in query
            or "다른 유물" in query
            or "추천" in query
        ):

            return "recommendation"

        # 일반 설명형
        return "description"