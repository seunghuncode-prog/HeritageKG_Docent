class GraphRetriever:

    def __init__(self, neo4j_client):

        self.neo4j = neo4j_client

    # ==================================================
    # 일반 retrieval
    # ==================================================

    def retrieve_subgraph(self, keyword):

        query = """
        MATCH (u:유물)-[r]-(n)

        WHERE
            u.title CONTAINS $keyword
            OR u.전시명칭 CONTAINS $keyword
            OR u.다른명칭 CONTAINS $keyword
            OR u.소장품번호 CONTAINS $keyword

        RETURN u, r, n
        LIMIT 20
        """

        return self.neo4j.run_query(
            query,
            {"keyword": keyword}
        )

    # ==================================================
    # 비슷한 유물 retrieval
    # ==================================================

    def retrieve_similar_artifacts(
        self,
        artifact_name
    ):

        query = """
        MATCH (u:유물)-[:재질로만들어짐]->(m)<-[:재질로만들어짐]-(other:유물)

        WHERE
            u.title CONTAINS $artifact_name
            AND u <> other

        RETURN DISTINCT other
        LIMIT 5
        """

        return self.neo4j.run_query(
            query,
            {"artifact_name": artifact_name}
        )