import weaviate
from weaviate import WeaviateClient
import os
from dotenv import load_dotenv
from src.logger1 import logger

load_dotenv()
WEAVIATE_CLUSTER_ENV = os.environ['WEAVIATE_CLUSTER_ENV']
WEAVIATE_API_KEY = os.environ['WEAVIATE_API_KEY']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
logger.info(f"Weaviate API key: {WEAVIATE_API_KEY}")
logger.info(f"WEAVIATE_CLUSTER_ENV: {WEAVIATE_CLUSTER_ENV}")

def get_weaviate_collection(collection_name : str = 'T2SQL'):
    logger.info("Entered get_weaviate_collection method")
    client: WeaviateClient = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_CLUSTER_ENV, auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
                                                headers={"X-OpenAI-Api-Key": OPENAI_API_KEY})

    collection = client.collections.get(collection_name)
    return collection

def retrieve_doc(query: str, alpha: float) -> str:
    logger.info("Entered retrieve_doc method")
    collection = get_weaviate_collection()
    response = collection.query.hybrid(query=query, alpha=alpha, limit=1)
    response = response.objects[0].properties['context'].split('\n')[1]
    return response

def add_documents(collection):
    pass