import streamlit as st
import os
import cv2
import numpy as np
import google.generativeai as genai
from moviepy import VideoFileClip
import imageio_ffmpeg as im_ffmpeg
import time
import whisper
import plotly.graph_objects as go
from fpdf import FPDF
import urllib.request
import ssl

# =========================================================
# 1. CẤU HÌNH & CSS STYLE
# =========================================================
st.set_page_config(page_title="Trợ lý chấm điểm", layout="wide", page_icon="🎓")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    .metric-card {
        background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 10px;
        padding: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-5px); box-shadow: 0 8px 15px rgba(0,0,0,0.1); }
    .metric-value { font-size: 28px; font-weight: 700; color: #1E88E5; margin-bottom: 5px; }
    .metric-label { font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .main-header { text-align: center; padding-bottom: 30px; color: #0D47A1; }
    .ai-box {
        background-color: #f0f7ff; border-left: 5px solid #0D47A1; padding: 20px;
        border-radius: 5px; margin-top: 20px; font-size: 16px; line-height: 1.6;
    }
    div.stButton > button {
        background-color: #0D47A1; color: white; border-radius: 8px; padding: 0.5rem 1rem;
        font-weight: bold; width: 100%; border: none;
    }
    div.stButton > button:hover { background-color: #1565C0; color: white; }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# 2. KHỞI TẠO HỆ THỐNG
# =========================================================

VISION_MODE = "None"
mp_face_mesh = None
mp_pose = None
face_cascade = None

try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh') and hasattr(mp.solutions, 'pose'):
        mp_face_mesh = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose
        VISION_MODE = "MediaPipe (AI Full)"
    else:
        raise ImportError("MediaPipe missing modules")
except Exception:
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        VISION_MODE = "OpenCV (Motion & Face)"
    except:
        VISION_MODE = "None"

HAS_VISION = True if VISION_MODE != "None" else False

HAS_AUDIO_AI = False
try:
    import librosa
    HAS_AUDIO_AI = True
except:
    HAS_AUDIO_AI = False

current_dir = os.path.dirname(os.path.abspath(__file__))
local_ffmpeg = os.path.join(current_dir, "ffmpeg.exe")
if os.path.exists(local_ffmpeg):
    os.environ["IMAGEIO_FFMPEG_EXE"] = local_ffmpeg
    os.environ["PATH"] += os.pathsep + current_dir

# --- API KEY ---
MY_GEMINI_KEY = "AIzaSyAczYDWBottda7vVZF5gVO8kx-PRpD2-WM" 
ai_ready = False
if MY_GEMINI_KEY:
    try:
        genai.configure(api_key=MY_GEMINI_KEY)
        ai_ready = True
    except: pass

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small", device="cpu")

# =========================================================
# 3. CÁC HÀM PHÂN TÍCH
# =========================================================

def analyze_video_comprehensive(video_path):
    if not HAS_VISION: return 0, 0
    cap = cv2.VideoCapture(video_path)
    eye_contact_frames = 0
    total_face_frames = 0
    prev_gray = None
    motion_accumulated = 0
    mesh = None
    if VISION_MODE == "MediaPipe (AI Full)":
        mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    
    count = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        if count % 10 == 0:
            total_face_frames += 1
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if prev_gray is not None:
                frame_delta = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                motion_pixels = cv2.countNonZero(thresh)
                h, w = gray.shape
                motion_ratio = motion_pixels / (h * w)
                if motion_ratio > 0.005: motion_accumulated += 1
            prev_gray = gray

            if VISION_MODE == "MediaPipe (AI Full)" and mesh:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = mesh.process(image_rgb)
                if results.multi_face_landmarks:
                    for fl in results.multi_face_landmarks:
                        h, w, _ = image.shape
                        face_2d = []
                        face_3d = []
                        for idx, lm in enumerate(fl.landmark):
                            if idx in [33, 263, 1, 61, 291, 199]:
                                x, y = int(lm.x * w), int(lm.y * h)
                                face_2d.append([x, y])
                                face_3d.append([x, y, lm.z])
                        f2d = np.array(face_2d, dtype=np.float64)
                        f3d = np.array(face_3d, dtype=np.float64)
                        cam = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])
                        success, rot, trans = cv2.solvePnP(f3d, f2d, cam, np.zeros((4,1)))
                        rmat, _ = cv2.Rodrigues(rot)
                        angles, _, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                        if -15 < angles[1]*360 < 15 and -12 < angles[0]*360 < 12:
                            eye_contact_frames += 1
            elif VISION_MODE == "OpenCV (Motion & Face)" and face_cascade:
                gray_f = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_f, 1.1, 4)
                if len(faces) > 0: eye_contact_frames += 1
        count += 1
    cap.release()
    if mesh: mesh.close()
    eye_score = (eye_contact_frames / total_face_frames * 100) if total_face_frames > 0 else 0
    gesture_ratio = (motion_accumulated / total_face_frames) if total_face_frames > 0 else 0
    gesture_score = min(gesture_ratio * 300, 100) 
    return int(eye_score), int(gesture_score)

def analyze_audio_prosody(audio_path):
    if not HAS_AUDIO_AI: return "N/A"
    try:
        y, sr = librosa.load(audio_path, duration=60) 
        rms = librosa.feature.rms(y=y)[0]
        variation = np.std(rms) / np.mean(rms)
        if variation < 0.3: return "Giọng đều"
        elif variation < 0.5: return "Bình thường"
        else: return "Truyền cảm (Tốt)"
    except: return "Không xác định"

def count_hesitations(text):
    words = text.lower().split()
    fillers = ['à', 'ờ', 'ừ', 'ừm', 'là', 'mà', 'kiểu', 'dạ']
    count = sum(1 for w in words if w in fillers)
    return count

def create_radar_chart(eye, gesture, voice_score, flow_score):
    categories = ['Giao tiếp mắt', 'Ngôn ngữ cơ thể', 'Giọng điệu', 'Sự trôi chảy']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[eye, gesture, voice_score, flow_score], theta=categories,
        fill='toself', name='Kết quả', line_color='#1E88E5', fillcolor='rgba(30, 136, 229, 0.3)'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, margin=dict(l=40, r=40, t=30, b=30), height=300)
    return fig

# --- HÀM TẠO PDF (DỰ PHÒNG 3 LỚP) ---
def download_font():
    font_path = "Roboto-Regular.ttf"
    if os.path.exists(font_path): return True
    
    # 3 nguồn tải font dự phòng
    urls = [
        "https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu4mxK.ttf",
        "https://cdnjs.cloudflare.com/ajax/libs/roboto-fontface/0.10.0/fonts/roboto/Roboto-Regular.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/apache/roboto/static/Roboto-Regular.ttf"
    ]
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=5) as response, open(font_path, 'wb') as out_file:
                out_file.write(response.read())
            if os.path.exists(font_path) and os.path.getsize(font_path) > 1000:
                return True
        except: continue
    return False

def remove_accents(input_str):
    if not input_str: return ""
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in input_str:
        if c in s1: s += s0[s1.index(c)]
        else: s += c
    return s

def create_pdf_report(topic, metrics, ai_feedback, transcript):
    has_font = download_font()
    
    class PDF(FPDF):
        def header(self):
            if has_font:
                self.set_font('Roboto', 'B', 15)
                self.cell(0, 10, 'BAO CAO DANH GIA (AI PRO)', 0, 1, 'C')
            else:
                self.set_font('Arial', 'B', 15)
                self.cell(0, 10, 'BAO CAO DANH GIA (KHONG DAU)', 0, 1, 'C')
            self.ln(5)

    pdf = PDF()
    
    if has_font:
        try:
            pdf.add_font('Roboto', '', "Roboto-Regular.ttf", uni=True)
            pdf.add_font('Roboto', 'B', "Roboto-Regular.ttf", uni=True)
        except: has_font = False
            
    pdf.add_page()
    
    def safe_write(txt, style='', size=12):
        if has_font:
            pdf.set_font("Roboto", style, size)
            return txt
        else:
            pdf.set_font("Arial", style, size)
            return remove_accents(txt)

    pdf.cell(0, 10, safe_write(f"Chu de: {topic}"), 0, 1)
    pdf.ln(5)

    pdf.cell(0, 10, safe_write("1. KET QUA PHAN TICH:", 'B'), 0, 1)
    col_width = 45
    row_height = 10
    metrics_data = [
        ("Giao tiep mat", f"{metrics['eye']}%"),
        ("Ngon ngu co the", f"{metrics['gesture']}/100"),
        ("Toc do noi", f"{metrics['wpm']} wpm"),
        ("Su ngap ngung", f"{metrics['hesitation']} lan"),
        ("Giong dieu", f"{metrics['voice']}")
    ]
    for row in metrics_data:
        pdf.cell(col_width, row_height, safe_write(row[0]), border=1)
        pdf.cell(col_width, row_height, safe_write(row[1]), border=1)
        pdf.ln(row_height)
    pdf.ln(10)

    pdf.cell(0, 10, safe_write("2. DANH GIA CHI TIET TU AI:", 'B'), 0, 1)
    clean_feedback = ai_feedback.replace('*', '').replace('#', '')
    pdf.multi_cell(0, 8, safe_write(clean_feedback, size=11))
    pdf.ln(10)
    
    pdf.cell(0, 10, safe_write("3. NOI DUNG (TRANSCRIPT):", 'B'), 0, 1)
    transcript_short = transcript[:3000] + "..." if len(transcript) > 3000 else transcript
    pdf.multi_cell(0, 6, safe_write(transcript_short, size=10))

    try: return bytes(pdf.output()) 
    except: return pdf.output(dest='S').encode('latin-1', 'ignore')

# --- HÀM GỌI GEMINI AN TOÀN (FIX 404 MODEL) ---
def generate_content_safe(prompt):
    # Danh sách model dự phòng (từ mới đến cũ)
    models_to_try = ['gemini-2.5-flash', 'gemini-2.5-flash-latest', 'gemini-pro', 'gemini-1.5-pro']
    
    last_error = None
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            last_error = e
            continue # Thử model tiếp theo
            
    # Nếu tất cả đều lỗi
    raise last_error

# =========================================================
# 4. GIAO DIỆN NGƯỜI DÙNG 
# =========================================================

st.markdown("<h1 class='main-header'>TRỢ LÝ CHẤM ĐIỂM THUYẾT TRÌNH AI<br><span style='font-size: 20px; font-weight: 300; color: #555'>Trợ lý chấm điểm thuyết trình thông minh ứng dụng trí tuệ nhân tạo</span></h1>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.markdown("THIẾT LẬP THÊM")
    with st.expander("📝 Thông tin nội dung", expanded=True):
        target_topic = st.text_input("Chủ đề chính", "Nhập chủ đề")
        keywords = st.text_area("Từ khóa trọng tâm", "khoa học, mô hình, kết quả, sáng tạo")
    st.markdown("---")
    def status_badge(label, status):
        color = "#4CAF50" if status else "#F44336"
        icon = "✅" if status else "❌"
        st.markdown(f"<span style='color:{color}; font-weight:bold'>{icon} {label}</span>", unsafe_allow_html=True)
    status_badge(f"AI quan sát: Sẵn sàng", HAS_VISION)
    status_badge("AI xử lí âm thanh: Sẵn sàng", HAS_AUDIO_AI)
    status_badge("Kết nối gemini: Sẵn sàng", ai_ready)
    st.markdown("---")
    st.caption("Developed by Quàng Văn Lợi")

with st.container():
    st.markdown("##### Tải lên video bài thuyết trình của bạn")
    uploaded_file = st.file_uploader("", type=['mp4', 'mov', 'avi'], help="Hỗ trợ định dạng MP4, AVI. Tốt nhất dưới 5 phút.")

if uploaded_file:
    col_vid, col_info = st.columns([1.5, 1], gap="large")
    with col_vid: st.video(uploaded_file)
    with col_info:
        st.markdown(f"**Tên video chấm:** `{uploaded_file.name}`")
        st.info("""
        **Hệ thống sẽ phân tích:**
        1. 👁️ **Giao tiếp mắt:** Mức độ tự tin của người trình bày.
        2. 👐 **Ngôn ngữ cơ thể:** Sự linh hoạt của tay và vai.
        3. 🔉 **Giọng điệu:** Cảm xúc và điểm nhấn trong giọng nói.
        4. 📝 **Nội dung:** Cấu trúc bài nói theo mục tiêu đề ra.
        """)
        analyze_btn = st.button("BẮT ĐẦU CHẤM ĐIỂM NGAY", width='stretch')

    if analyze_btn:
        if not ai_ready:
            st.error("Vui lòng kiểm tra API Key Gemini!")
            st.stop()
        
        metrics = {"duration": 0, "wpm": 0, "eye": 0, "gesture": 0, "voice": "N/A", "hesitation": 0}
        temp_v, temp_a = "temp.mp4", "temp.mp3"
        
        with st.status("Đang phân tích video chuyên sâu...", expanded=True) as status:
            st.write("Đang trích xuất dữ liệu video...")
            with open(temp_v, "wb") as f: f.write(uploaded_file.getbuffer())
            
            st.write("Đang tách và xử lý âm thanh...")
            video = VideoFileClip(temp_v)
            metrics['duration'] = video.duration
            video.audio.write_audiofile(temp_a, logger=None)
            video.close()
            
            st.write("Đang quan sát ánh mắt và cử chỉ...")
            if HAS_VISION:
                metrics['eye'], metrics['gesture'] = analyze_video_comprehensive(temp_v)
            
            st.write("Đang đo lường cảm xúc giọng nói...")
            if HAS_AUDIO_AI:
                metrics['voice'] = analyze_audio_prosody(temp_a)
                
            st.write("Đang phân tích nội dung thuyết trình...")
            model = load_whisper_model()
            prompt = f"Thuyết trình về {target_topic}. Từ khóa: {keywords}."
            result = model.transcribe(temp_a, language='vi', initial_prompt=prompt, fp16=False)
            transcript = result['text']
            
            metrics['wpm'] = int(len(transcript.split()) / (metrics['duration']/60)) if metrics['duration']>0 else 0
            metrics['hesitation'] = count_hesitations(transcript)
            
            st.write("AI đang tổng hợp đánh giá cuối cùng...")
            gesture_text = "Rất linh hoạt, sử dụng tay nhiều" if metrics['gesture'] > 80 else ("Có cử động nhưng còn ít" if metrics['gesture'] > 50 else "Đứng im, thiếu cử chỉ tay")
            
            final_prompt = f"""
            Vai trò: Giám khảo chuyên nghiệp cuộc thi thuyết trình.
            Dữ liệu:
            - Chủ đề: {target_topic}
            - Văn bản nói: "{transcript}"
            - Tốc độ nói: {metrics['wpm']} từ/phút.
            - Số lần ậm ừ: {metrics['hesitation']} lần.
            - Tông giọng: {metrics['voice']}.
            - Giao tiếp mắt (Eye Contact): {metrics['eye']}/100.
            - Ngôn ngữ cơ thể (Body Language): {metrics['gesture']}/100 ({gesture_text}).
            
            Yêu cầu output định dạng Markdown chuyên nghiệp:
            1. **NHẬN XÉT CHUNG**: 2 câu tóm tắt thần thái, xưng tôi/bạn.
            2. **TỔNG ĐIỂM (0-100)**: Con số cụ thể.
            3. **ĐIỂM CỤ THỂ TIÊU CHÍ**:
                - Nội dung: /35 điểm
                - Giọng điệu và ngữ điệu: /15 điểm
                - Ngôn ngữ cơ thể: /15 điểm
                - Giao tiếp mắt: /10 điểm
                - Lưu loát: /10 điểm
                - Tính thuyết phục: /15 điểm
            4. **ĐIỂM MẠNH & ĐIỂM YẾU**: Dạng bullet point ngắn gọn.
            5. **LỜI KHUYÊN CẢI THIỆN**: 3 ý chính.
            """
            try:
                # GỌI HÀM AN TOÀN ĐÃ ĐỊNH NGHĨA Ở TRÊN
                ai_feedback = generate_content_safe(final_prompt)
                status.update(label="✅ Phân tích hoàn tất!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Lỗi kết nối AI (Đã thử mọi model): {e}")
                status.update(label="❌ Lỗi AI", state="error")
                st.stop()

        if os.path.exists(temp_a): os.remove(temp_a)
        if os.path.exists(temp_v): os.remove(temp_v)

        st.divider()
        st.markdown("### 📊 Kết quả phân tích")
        m1, m2, m3, m4 = st.columns(4)
        def render_card(col, label, value, unit=""):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}<span style="font-size:16px; color:#888">{unit}</span></div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m1: render_card(st, "Giao tiếp mắt", metrics['eye'], "%")
        with m2: render_card(st, "Ngôn ngữ cơ thể", metrics['gesture'], "/100")
        with m3: render_card(st, "Tốc độ nói", metrics['wpm'], " wpm")
        with m4: render_card(st, "Sự ngập ngừng", metrics['hesitation'], " lần")

        st.divider()
        c_left, c_right = st.columns([1.8, 1.2]) 
        with c_left:
            st.markdown("#### Đánh giá từ chuyên gia")
            st.markdown(f"""<div class="ai-box">{ai_feedback}</div>""", unsafe_allow_html=True)
            
        with c_right:
            st.markdown("#### Biểu đồ Kỹ năng")
            voice_map = {"Giọng đều": 50, "Bình thường": 75, "Truyền cảm (Tốt)": 95, "N/A": 0, "Không xác định": 50}
            voice_score = voice_map.get(metrics['voice'], 60)
            flow_score = max(0, 100 - (metrics['hesitation'] * 5))
            fig = create_radar_chart(metrics['eye'], metrics['gesture'], voice_score, flow_score)
            st.plotly_chart(fig, width='stretch')
            with st.expander("📄 Xem Transcript"): st.write(transcript)
        
        st.divider()
        pdf_bytes = create_pdf_report(target_topic, metrics, ai_feedback, transcript)
        
        if download_font() == False:
            st.warning("⚠️ Cảnh báo: Mạng đang chặn tải Font. PDF được xuất dưới dạng KHÔNG DẤU để tránh lỗi.")
            
        if pdf_bytes:
            st.download_button(label="📥 Tải Báo cáo PDF", data=pdf_bytes, file_name="baocao_danhgia.pdf", mime="application/pdf")