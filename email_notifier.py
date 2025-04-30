import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import socket

# Configure logging
logger = logging.getLogger(__name__)

# Email server configuration - UPDATE THESE VALUES
EMAIL_SERVER = "smtp.qq.com"  # Your SMTP server
EMAIL_PORT = 587  # Your SMTP port
EMAIL_USERNAME = "1076896291@qq.com"  # Your email username/address
EMAIL_PASSWORD = "hwsrfpzatyprffie"  # Your email password
EMAIL_FROM = "1076896291@qq.com"  # Sender email address

def send_notification_email(recipient_email, subject, body):
    """
    Send a notification email
    
    Args:
        recipient_email: Email address to send to
        subject: Email subject
        body: Email body
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not recipient_email:
        logger.warning("No recipient email provided")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Attach body
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to server and send email
        server = smtplib.SMTP(EMAIL_SERVER, EMAIL_PORT)
        server.starttls()  # Use TLS
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email sent to {recipient_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        return False 