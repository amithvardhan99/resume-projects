import numpy as np
import pandas as pd
import os
import json

from imap_tools import MailBox
from dotenv import load_dotenv
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

IMAP_HOST = os.getenv("IMAP_HOST")
IMAP_USER = os.getenv("IMAP_USER")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD")
IMAP_FOLDER = "INBOX"

class ChatState(TypedDict):
    messages: list

def connect():
    """Connect to my Gmail account using this function"""
    mail_box = MailBox(IMAP_HOST)
    with mail_box.login(IMAP_USER,IMAP_PASSWORD,initial_folder="INBOX") as m:
        print("Connected successfully")
        return m

if __name__ == "__main__":
    print(connect())

