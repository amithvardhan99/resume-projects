import os
import json
import pickle
import imaplib
import email
from email.header import decode_header
from typing import TypedDict

# Google OAuth Libraries
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# LangChain
from langchain_core.tools import tool


from dotenv import load_dotenv


load_dotenv()

class ChatState(TypedDict):
    messages: list

IMAP_HOST = os.getenv("IMAP_HOST")
IMAP_USER = os.getenv("IMAP_USER")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD")
IMAP_FOLDER = "INBOX"

print("HOST: ",IMAP_HOST)
print("USER:",IMAP_USER)

SCOPES = ["https://mail.google.com/"]

TOKEN_FILE = "token.pickle"
CREDENTIALS_FILE = "credentials.json"


# Fetching the OAuth credentials
def get_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as fp:
            creds = pickle.load(fp)
    if creds:
        if creds.expired:
            if creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception:
                    creds = None
            else:
                creds = None

    if not creds:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "wb") as fp:
            pickle.dump(creds, fp)
    return creds


# Using the OAuth credentials, Gmail's IMAP host and the username to login to the Gmail IMAP
def connect():
    creds = get_credentials()
    mail = imaplib.IMAP4_SSL(IMAP_HOST)
    authorisation_string = f"user={IMAP_USER}\1auth=Bearer {creds.token}\1\1"
    mail.authenticate("XOAUTH2", lambda x: authorisation_string)
    return mail

# Write a function to list all the unread messages from the GMail Inbox, list them; and make it a tool
@tool
def list_unread_emails():
    """Returns all the unread emails with UID, Subject, Date, Sender from the inbox"""
    print("List Unread Emails tool called")
    conn = connect()
    conn.select("INBOX")
    status, data = conn.search(None, "UNSEEN")


    list_of_emails = []

    for i in data[0].split():
        status, msg = conn.fetch(i,"(BODY.PEEK[])")
        if status != "OK":
            continue
        raw_email = msg[0][1]
        email_message = email.message_from_bytes(raw_email)

        # Decode the subject of the email
        subject, encoding = decode_header(email_message.get("Subject"))[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding or "utf-8", errors="ignore")

        sender = email_message.get("From")
        date = email_message.get("Date")

        list_of_emails.append({
            "uid": i.decode(), #Message serial number according to IMAP
            "date": date,#.astimezone().strftime("%Y-%m-%d %H:%M"),
            "subject": subject,
            "sender": sender
        })

    return json.dumps(list_of_emails, indent=2)

# Function that summarises the content of an email, given its UID
def prompt_for_summarise_email(uid):
    """Frames a prompt required to summarise the content from an email given its UID"""
    conn = connect()
    conn.select("INBOX")
    # status, data = conn.search(None, "UNSEEN")

    status, msg = conn.fetch(uid, "(BODY.PEEK[])")
    if status != "OK":
        return None
    raw_email = msg[0][1]
    email_message = email.message_from_bytes(raw_email)

    # Decode the subject of the email
    subject, encoding = decode_header(email_message.get("Subject"))[0]
    if isinstance(subject, bytes):
        subject = subject.decode(encoding or "utf-8", errors="ignore")

    sender = email_message.get("From")
    date = email_message.get("Date")

    body = ""
    if email_message.is_multipart():
        for part in email_message.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition"))
            if (content_type == "text/plain" and "attachment" not in disposition):
                payload = part.get_payload(decode=True)
                if payload:
                    body += payload.decode(errors="ignore") + "\n"
                    # break
    else:
        payload = email_message.get_payload(decode=True)
        if payload:
            body = payload.decode(errors="ignore")

    prompt = (
        f"Summarise this email concisely:\n"
        f"Subject: {subject}\n"
        f"Date: {date}\n"
        f"Sender: {sender}\n"
        f"{body}"
    )

    return prompt

# Final Summarisation tool
@tool
def summarise_email(uid):
    """Returns the summary of an email given its UID"""
    print("Summarise Email tool called")
    prompt = prompt_for_summarise_email(uid)
    return raw_llm.invoke(prompt).content