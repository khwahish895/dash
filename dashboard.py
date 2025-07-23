import streamlit as st
import psutil
import pywhatkit as kit
import smtplib
from twilio.rest import Client
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import yagmail
import requests
import time
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
from dotenv import load_dotenv
from urllib.parse import quote
from linkedin_api import Linkedin
import paramiko
import json
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
import streamlit.components.v1 as components

# Load environment variables
load_dotenv()

# Initialize services
twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
yag = yagmail.SMTP(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
linkedin_client = Linkedin(os.getenv("LINKEDIN_EMAIL"), os.getenv("LINKEDIN_PASSWORD"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Chrome options for selenium
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")

# App configuration
st.set_page_config(page_title="Ultimate Automation Suite", layout="wide")
st.title("🤖 Ultimate Python Automation Suite")
st.markdown("A comprehensive collection of automation tasks with image processing capabilities.")

# ============ MAIN TABS ============
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📞 Communication", 
    "🌐 Web Tools", 
    "🖼️ Image Processing", 
    "📊 Machine Learning", 
    "🧠 AI Tools"
])

# ============ COMMUNICATION TOOLS ============
with tab1:
    st.header("📞 Communication Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💬 WhatsApp Messaging")
        whatsapp_number = st.text_input("Phone Number (+countrycode)", key="whatsapp_num")
        whatsapp_msg = st.text_area("Message", key="whatsapp_msg")
        if st.button("Send WhatsApp Message"):
            if whatsapp_number and whatsapp_msg:
                kit.sendwhatmsg_instantly(whatsapp_number, whatsapp_msg, wait_time=15)
                st.success("WhatsApp message scheduled!")
            else:
                st.error("Please enter both number and message")

        st.subheader("📱 SMS via Twilio")
        sms_to = st.text_input("To Phone Number", key="sms_to")
        sms_msg = st.text_area("SMS Message", key="sms_msg")
        if st.button("Send SMS"):
            try:
                message = twilio_client.messages.create(
                    body=sms_msg,
                    from_=os.getenv("TWILIO_PHONE_NUMBER"),
                    to=sms_to
                )
                st.success(f"SMS sent! SID: {message.sid}")
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        st.subheader("📧 Email Sender")
        email_to = st.text_input("To Email", key="email_to")
        email_subj = st.text_input("Subject", key="email_subj")
        email_body = st.text_area("Message", key="email_body")
        if st.button("Send Email"):
            try:
                yag.send(
                    to=email_to,
                    subject=email_subj,
                    contents=email_body
                )
                st.success("Email sent successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

        st.subheader("📞 Phone Call via Twilio")
        call_to = st.text_input("Number to call", key="call_to")
        call_msg = st.text_area("Message to speak", key="call_msg")
        if st.button("Make Call"):
            try:
                call = twilio_client.calls.create(
                    twiml=f'<Response><Say>{call_msg}</Say></Response>',
                    to=call_to,
                    from_=os.getenv("TWILIO_PHONE_NUMBER")
                )
                st.success(f"Call initiated! SID: {call.sid}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.subheader("💼 LinkedIn Messaging")
    if st.checkbox("Show LinkedIn Options"):
        try:
            connections = linkedin_client.get_profile_connections()
            connections = [f"{c['firstName']} {c['lastName']}" for c in connections]
            recipient = st.selectbox("Select Connection", connections)
            message = st.text_area("Your Message", key="linkedin_msg")
            if st.button("Send LinkedIn Message"):
                recipient_id = [c['entityUrn'].split(':')[-1] for c in linkedin_client.get_profile_connections() 
                              if f"{c['firstName']} {c['lastName']}" == recipient][0]
                linkedin_client.send_message(recipient_id, message)
                st.success("Message sent successfully!")
        except Exception as e:
            st.error(f"LinkedIn Error: {e}")

# ============ WEB TOOLS ============
with tab2:
    st.header("🌐 Web Automation Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Google Search Scraper")
        search_query = st.text_input("Search Term", key="search_term")
        num_results = st.slider("Results to fetch", 1, 20, 5)
        if st.button("Search Google"):
            try:
                results = []
                for j in search(search_query, num_results=num_results):
                    results.append(j)
                st.dataframe(pd.DataFrame(results, columns=["Results"]))
            except Exception as e:
                st.error(f"Search error: {e}")

        st.subheader("🖥️ Remote Server Management")
        server_ip = st.text_input("Server IP", key="server_ip")
        server_user = st.text_input("Username", key="server_user")
        server_pass = st.text_input("Password", type="password", key="server_pass")
        server_cmd = st.text_input("Command to run", key="server_cmd")
        if st.button("Execute Remotely"):
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(server_ip, username=server_user, password=server_pass)
                stdin, stdout, stderr = ssh.exec_command(server_cmd)
                st.code(stdout.read().decode())
                ssh.close()
            except Exception as e:
                st.error(f"SSH Error: {e}")

    with col2:
        st.subheader("📍 Location Services")
        components.html("""
        <script>
        navigator.geolocation.getCurrentPosition(
            pos => document.getElementById('location').textContent = 
                `Lat: ${pos.coords.latitude.toFixed(4)}, Lon: ${pos.coords.longitude.toFixed(4)}`,
            err => document.getElementById('location').textContent = err.message
        );
        </script>
        <p id="location">Getting location...</p>
        """, height=100)

        st.subheader("🌐 Website Scraper")
        scrape_url = st.text_input("URL to scrape", key="scrape_url")
        if st.button("Fetch Page"):
            try:
                res = requests.get(scrape_url)
                soup = BeautifulSoup(res.text, 'html.parser')
                st.download_button(
                    "Download HTML",
                    str(soup),
                    file_name="scraped_page.html"
                )
                st.success(f"Found {len(soup.find_all())} elements")
            except Exception as e:
                st.error(f"Error: {e}")

# ============ IMAGE PROCESSING ============
with tab3:
    st.header("🖼️ Image Processing Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎨 Digital Image Creator")
        img_type = st.selectbox("Image Type", ["Abstract", "Geometric", "Landscape", "Portrait"])
        img_width = st.slider("Width", 300, 1200, 600)
        img_height = st.slider("Height", 300, 1200, 400)
        
        if st.button("Generate Image"):
            img = Image.new("RGB", (img_width, img_height), "#4285F4")
            draw = ImageDraw.Draw(img)
            
            if img_type == "Abstract":
                for _ in range(50):
                    x1, y1 = np.random.randint(0, img_width), np.random.randint(0, img_height)
                    x2, y2 = np.random.randint(0, img_width), np.random.randint(0, img_height)
                    color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                    draw.line([x1,y1,x2,y2], fill=color, width=2)
            
            elif img_type == "Geometric":
                for _ in range(15):
                    pts = [(np.random.randint(0,img_width), np.random.randint(0,img_height)) for _ in range(4)]
                    color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                    draw.polygon(pts, fill=color)
            
            st.image(img, use_column_width=True)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button("Download", buf.getvalue(), "image.png", "image/png")

    with col2:
        st.subheader("😊 Face Swapping")
        st.warning("Requires OpenCV face detection models")
        
        face1 = st.file_uploader("Face Source", type=["jpg","png"], key="face1")
        face2 = st.file_uploader("Face Target", type=["jpg","png"], key="face2")
        
        if st.button("Swap Faces") and face1 and face2:
            try:
                # Download models if not present
                if not os.path.exists("deploy.prototxt"):
                    with open("deploy.prototxt", "wb") as f:
                        f.write(requests.get("https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt").content)
                
                if not os.path.exists("res10_300x300_ssd_iter_140000.caffemodel"):
                    with open("res10_300x300_ssd_iter_140000.caffemodel", "wb") as f:
                        f.write(requests.get("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel").content)
                
                # Detect faces
                net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
                
                def get_face(image):
                    image = np.array(Image.open(image))
                    (h, w) = image.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                    net.setInput(blob)
                    detections = net.forward()
                    box = detections[0,0,0,3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    return image[y1:y2, x1:x2], (x1, y1, x2, y2)
                
                src_face, _ = get_face(face1)
                tgt_face, (tx1, ty1, tx2, ty2) = get_face(face2)
                
                # Resize and blend
                resized = cv2.resize(src_face, (tx2-tx1, ty2-ty1))
                mask = np.zeros(resized.shape[:2], dtype=np.uint8)
                center = ((tx1+tx2)//2, (ty1+ty2)//2)
                radius = min(tx2-tx1, ty2-ty1)//2
                cv2.circle(mask, center, radius, 255, -1)
                
                target_img = np.array(Image.open(face2))
                output = cv2.seamlessClone(resized, target_img, mask, center, cv2.NORMAL_CLONE)
                
                st.image(output, channels="BGR", use_column_width=True)
                buf = io.BytesIO()
                Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
                st.download_button("Download", buf.getvalue(), "face_swap.png", "image/png")
                
            except Exception as e:
                st.error(f"Face swap error: {e}")

# ============ MACHINE LEARNING ============
with tab4:
    st.header("📊 Machine Learning Tools")
    
    st.subheader("💰 Salary Predictor")
    if not os.path.exists("salary_model.pkl"):
        df = pd.DataFrame({
            'Experience': [1,2,3,4,5,6,7,8,9,10],
            'Salary': [30000,35000,40000,45000,50000,55000,60000,65000,70000,75000]
        })
        model = LinearRegression().fit(df[['Experience']], df['Salary'])
        joblib.dump(model, "salary_model.pkl")
    
    exp = st.slider("Years Experience", 0, 30, 5)
    salary = joblib.load("salary_model.pkl").predict([[exp]])[0]
    st.metric("Predicted Salary", f"${int(salary):,}")
    
    st.subheader("🎓 GPA Calculator")
    marks = st.number_input("Your Marks", min_value=0)
    total = st.number_input("Total Marks", min_value=1)
    if total > 0:
        st.metric("GPA Score", round((marks/total)*10, 2))

# ============ AI TOOLS ============
with tab5:
    st.header("🧠 AI Tools")
    
    st.subheader("🤖 Gemini AI Assistant")
    prompt = st.text_area("Ask me anything", key="gemini_prompt")
    if st.button("Get Answer"):
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            st.markdown(response.text)
        except Exception as e:
            st.error(f"AI Error: {e}")

# ============ SIDEBAR ============
st.sidebar.header("🛠️ Setup Instructions")
st.sidebar.code("""pip install streamlit pywhatkit twilio selenium 
beautifulsoup4 yagmail opencv-python pillow numpy pandas 
scikit-learn google-generativeai paramiko""")

