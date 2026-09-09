class QueryRouter:

    def route(self, query):

        if "알려줘" in query and "시대" in query:
            return "verification"

        if "다른 유물" in query:
            return "graph"

        return "semantic"