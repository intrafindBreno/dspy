import dspy
from opensearchpy import OpenSearch

from typing import Union, List


class OpenSearchRM(dspy.Retrieve):
    def __init__(self, k=3):
        super().__init__(k=k)
        self._opensearch = OpenSearch(
            hosts=[{"host": "hal9000", "port": 9200}], http_compress=True
        )

    def forward(self, query_or_queries: Union[str, List[str]], **kwargs) -> dspy.Prediction:
        """Search with OpenSearch for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        docs = []
        for query in queries:
            result = self._opensearch.search(
                index="dip-index",
                body={
                    "_source": False,
                    "query": {
                        "nested": {
                            "path": "passages",
                            "inner_hits": {"_source": ["passages.content"]},
                            "query": {
                                "bool": {
                                    "must": [
                                        {
                                            "neural": {
                                                "passages.vectors": {
                                                    "query_text": query,
                                                    "model_id": "iLQlhosB7Fuz8cJB3yhN",
                                                    "k": kwargs["k"] | self.k,
                                                }
                                            }
                                        }
                                    ]
                                }
                            },
                        }
                    },
                }
            )
            for hit in result["hits"]["hits"]:
                docs.append(hit["inner_hits"]["passages"]["hits"]["hits"][0]["_source"]["content"])
        return dspy.Prediction(passages=docs)
