import os
import smtplib
import argparse
from email.message import EmailMessage


def load_email_credentials_from_key(file_path=None):
    creds = {}
    if not os.path.exists(file_path):
        print(f"Warning: key file '{file_path}' not found.")
        return creds
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                creds[k.strip()] = v.strip()
    return creds


def send_email(args):
    msg = EmailMessage()
    msg["Subject"] = args.subject
    msg["From"] = args.email_sender
    msg["To"] = args.email_receiver
    msg.set_content(args.body)
    msg.add_alternative(args.body, subtype="html")  # Add HTML body

    with smtplib.SMTP(args.smtp_server, args.smtp_port) as s:
        s.starttls()
        s.login(args.email_sender, args.email_password)
        s.send_message(msg)


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--email_keyfile", type=str, default=".email_cred.key", help="Path to the .key file with email credentials.")
    parser.add_argument("--email_sender", type=str, default=None, help="Sender email address")
    parser.add_argument("--email_receiver", type=str, default=None, help="Receiver email address")
    parser.add_argument("--email_password", type=str, default=None, help="App password for sender")
    parser.add_argument("--smtp_server", type=str, default=None, help="SMTP server address for sending email")
    parser.add_argument("--smtp_port", type=int, default=None, help="SMTP server port number")

    parser.add_argument("--subject", type=str, default="Test Title", help="Email subject")
    parser.add_argument("--body", type=str, default="Test Body", help="Email body")
    parser.add_argument("--file", type=str, default=None, help="File to attach as email body")
    args = parser.parse_args()
    # fmt: on

    # Load email_credentials and smtp configs
    creds = load_email_credentials_from_key(args.email_keyfile)
    args.email_sender = args.email_sender or creds.get("EMAIL_SENDER")
    args.email_receiver = args.email_receiver or creds.get("EMAIL_RECEIVER")
    args.email_password = args.email_password or creds.get("EMAIL_PASSWORD")
    args.smtp_server = args.smtp_server or creds.get("SMTP_SERVER")
    args.smtp_port = args.smtp_port or creds.get("SMTP_PORT")
    if args.smtp_port is not None:
        args.smtp_port = int(args.smtp_port)

    if not args.email_sender or not args.email_receiver or not args.email_password:
        raise ValueError(f"Email credentials not provided and not found in {args.email_keyfile} file")  # fmt: skip
    if not args.smtp_server or not args.smtp_port:
        raise ValueError(f"SMTP configs for sending email not provided and not found in {args.email_keyfile} file")  # fmt: skip

    # Email contents
    if args.file is not None:
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if args.body is not None:
            args.body += "\n" + "=" * 20 + "\n"
        args.body = (args.body or "") + content
        args.body = f"<pre>{args.body}</pre>"  # display logs as in terminal

    # Send email
    if args.subject is None:
        print("Send email without title")
    if args.body is None:
        print("Send email with empty body")

    send_email(args)
