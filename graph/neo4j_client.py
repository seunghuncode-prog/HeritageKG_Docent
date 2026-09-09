import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

class Neo4jClient:

    def __init__(self):

        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(
                os.getenv("NEO4J_USER"),
                os.getenv("NEO4J_PASSWORD")
            )
        )

    def run_query(self, query, params=None):

        with self.driver.session() as session:

            result = session.run(
                query,
                params or {}
            )

            return [
                record.data()
                for record in result
            ]