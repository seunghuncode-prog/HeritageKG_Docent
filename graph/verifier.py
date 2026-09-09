class FactVerifier:
    def __init__(self, neo4j_client):
        self.neo4j = neo4j_client

    def verify_period(self, relic, claimed_period):

        query = """
        MATCH (r:Relic {name:$relic})-[:BELONGS_TO]->(p)
        RETURN p.name AS actual_period
        """

        result = self.neo4j.run_query(query, {
            "relic": relic
        })

        if not result:
            return {
                "verified": False,
                "message": "유물을 찾을 수 없습니다."
            }

        actual = result[0]["actual_period"]

        return {
            "verified": actual == claimed_period,
            "actual_period": actual
        }