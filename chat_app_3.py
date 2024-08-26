import streamlit as st
import os
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType

# CLUSTER_ENDPOINT = "YOUR_CLUSTER_ENDPOINT"
# TOKEN = "YOUR_CLUSTER_TOKEN"

# 1. Set up a Milvus client
client = MilvusClient(
    uri=os.environ.get("CLUSTER_ENDPOINT"),
    token=os.environ.get("MILVUS_TOKEN")
)

load_dotenv()