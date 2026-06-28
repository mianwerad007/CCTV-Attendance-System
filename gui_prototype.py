import cv2
import numpy as np
import sqlite3
import time
from datetime import datetime
import threading
import queue
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis
from db_helper import DB_PATH

class HighEfficiencyCCTVEngine:
    def __init__(self, root):
        self.root = root
        self.root.title("TrustSafe CCTV Attendance System - High-Efficiency Core")
        self.root.geometry("1450x720")
        self.root.configure(bg="#121214")
        
        # Performance Settings
        self.AI_INPUT_SIZE = (320, 320)  # Downscaling size for instant face lock
        self.MATCH_THRESHOLD = 0.58       # Recognition confidence cutoff
        self.FRAME_SKIP = 3              # Process every 3rd frame for AI calculation
        
        # Asynchronous Queue Buffers
        self.frame_queue = queue.Queue(maxsize=2)
        self.log_queue = queue.Queue()
        self.is_running = False
        
        # Load Core AI Pipeline
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=self.AI_INPUT_SIZE)
        
        self.known_faces = self.load_known_faces()
        self.setup_ui()
        self.refresh_logs()
        
        # Start Background Database Sync Worker
        threading.Thread(target=self.database_sync_worker, daemon=True).start()

    def load_known_faces(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT EmployeeID, FaceEncoding FROM Employees WHERE Status = 'Active'")
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            rows = []
        conn.close()
        
        known_faces = []
        for emp_id, db_blob in rows:
            if db_blob is not None:
                db_embedding = np.frombuffer(db_blob, dtype=np.float32)
                known_faces.append((emp_id, db_embedding))
        return known_faces

    def setup_ui(self):
        self.video_frame = tk.Frame(self.root, width=640, height=480, bg="#1a1a1e", highlightbackground="#2d2d34", highlightthickness=1)
        self.video_frame.pack(side=tk.LEFT, padx=15, pady=20)
        
        self.video_label = tk.Label(self.video_frame, bg="#1a1a1e")
        self.video_label.pack()

        self.right_panel = tk.Frame(self.root, bg="#121214")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=20)

        self.control_frame = tk.LabelFrame(self.right_panel, text=" Channel Configuration Link ", bg="#1a1a1e", fg="#007acc", font=("Arial", 10, "bold"), highlightthickness=0)
        self.control_frame.pack(fill=tk.X, pady=(0, 15), ipady=5, ipadx=10)

        tk.Label(self.control_frame, text="RTSP Stream or Webcam ID:", bg="#1a1a1e", fg="#a0a0a5").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.rtsp_entry = tk.Entry(self.control_frame, width=45, bg="#2d2d34", fg="#ffffff", insertbackground="white", borderwidth=0)
        self.rtsp_entry.insert(0, "0")
        self.rtsp_entry.grid(row=0, column=1, padx=10, pady=10)

        self.btn_action = tk.Button(self.control_frame, text="START STREAM ENGINE", command=self.toggle_engine, bg="#007acc", fg="#ffffff", borderwidth=0, font=("Arial", 9, "bold"), cursor="hand2")
        self.btn_action.grid(row=0, column=2, padx=10, pady=10)

        tk.Label(self.right_panel, text="Real-Time Active Core Logs Dashboard", bg="#121214", fg="#ffffff", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 5))

        # Re-engineered to exactly display your 9 unique dashboard requirements
        columns_schema = ("Sr", "EmpCode", "Name", "FatherName", "Department", "Designation", "TimeIn", "TimeOut", "TotalHours")
        self.tree = ttk.Treeview(self.right_panel, columns=columns_schema, show="headings")
        
        self.tree.heading("Sr", text="Sr#")
        self.tree.heading("EmpCode", text="Emp Code")
        self.tree.heading("Name", text="Name")
        self.tree.heading("FatherName", text="Father Name")
        self.tree.heading("Department", text="Department")
        self.tree.heading("Designation", text="Designation")
        self.tree.heading("TimeIn", text="Time In (Date)")
        self.tree.heading("TimeOut", text="Time Out (Date)")
        self.tree.heading("TotalHours", text="Total Duty Hours")
        
        column_widths = {"Sr": 45, "EmpCode": 85, "Name": 115, "FatherName": 115, "Department": 115, "Designation": 115, "TimeIn": 145, "TimeOut": 145, "TotalHours": 110}
        for col in columns_schema:
            self.tree.column(col, width=column_widths[col], anchor="center")
            
        self.tree.pack(fill=tk.BOTH, expand=True)

    def toggle_engine(self):
        if not self.is_running:
            self.is_running = True
            self.btn_action.config(text="STOP STREAM ENGINE", bg="#d9534f")
            self.known_faces = self.load_known_faces()
            
            target_source = self.rtsp_entry.get().strip()
            stream_source = int(target_source) if target_source.isdigit() else target_source
            
            # Start parallel asymmetric pipeline workers
            threading.Thread(target=self.video_stream_producer, args=(stream_source,), daemon=True).start()
            threading.Thread(target=self.ai_inference_consumer, daemon=True).start()
        else:
            self.is_running = False
            self.btn_action.config(text="START STREAM ENGINE", bg="#007acc")
            self.video_label.config(image="")

    def video_stream_producer(self, source):
        """Thread 1: Dedicated to raw frame acquisition with zero buffering latency."""
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self.is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Push the freshest frame to the queue, discarding old frames
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

        cap.release()

    def ai_inference_consumer(self):
        """Thread 2: Asynchronous AI analysis loop running with Temporal Smoothing Filter."""
        frame_counter = 0
        tracking_memory = {}
        STABILIZATION_FRAMES = 6  # Rolling window size to eliminate jitter
        
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            frame_counter += 1
            if frame_counter % self.FRAME_SKIP != 0:
                self.root.after(0, self.update_ui_canvas, frame)
                continue

            h, w = frame.shape[:2]
            small_frame = cv2.resize(frame, (w // 2, h // 2))
            
            faces = self.app.get(small_frame)
            
            for i, face in enumerate(faces):
                bbox = (face.bbox * 2).astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                live_embedding = face.normed_embedding

                best_match = "Unknown"
                highest_sim = 0.0

                if self.known_faces:
                    for emp_id, db_embedding in self.known_faces:
                        if live_embedding.shape == db_embedding.shape:
                            similarity = float(np.dot(live_embedding, db_embedding))
                            if similarity > highest_sim:
                                highest_sim = similarity
                                if similarity >= self.MATCH_THRESHOLD:
                                    best_match = emp_id

                # --- Temporal Stabilization Filter Logic ---
                if best_match != "Unknown":
                    tracking_memory[i] = {"name": best_match, "frames_left": STABILIZATION_FRAMES}
                    self.log_queue.put((best_match, highest_sim * 100))
                else:
                    if i in tracking_memory and tracking_memory[i]["frames_left"] > 0:
                        best_match = tracking_memory[i]["name"]
                        tracking_memory[i]["frames_left"] -= 1
                    else:
                        if i in tracking_memory:
                            del tracking_memory[i]

                # Stable UI Tagging
                color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)
                label = f"{best_match}" if best_match != "Unknown" else "Analyzing..."
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if len(faces) == 0:
                tracking_memory.clear()

            self.root.after(0, self.update_ui_canvas, frame)

    def update_ui_canvas(self, frame):
        """Thread 3 (Main): Render the live image instantly onto the screen canvas."""
        try:
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv2_image).resize((640, 480))
            img_tk = ImageTk.PhotoImage(image=pil_img)
            if self.is_running:
                self.video_label.imgtk = img_tk
                self.video_label.config(image=img_tk)
        except Exception:
            pass

    def database_sync_worker(self):
        """Thread 4: Isolated SQLite logging agent. Prevents file locks from freezing UI."""
        while True:
            try:
                employee_id, confidence = self.log_queue.get(block=True)
                now = datetime.now()
                current_date = now.strftime("%Y-%m-%d")
                current_time_str = now.strftime("%H:%M:%S")
                timestamp_with_date = f"{current_time_str} ({current_date})"

                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                cursor.execute("SELECT ID, TimeIn, TimeOut FROM Attendance WHERE EmployeeID = ? AND LogDate = ?", (employee_id, current_date))
                existing_record = cursor.fetchone()

                if not existing_record:
                    cursor.execute('''
                        INSERT INTO Attendance (EmployeeID, LogDate, TimeIn, TimeOut, TotalDutyHours, CameraID, Confidence)
                        VALUES (?, ?, ?, '', '00:00', 1, ?)
                    ''', (employee_id, current_date, timestamp_with_date, float(confidence)))
                    conn.commit()
                else:
                    record_id, time_in_val, time_out_val = existing_record
                    clean_time_in = time_in_val.split(" ")[0]
                    time_in_obj = datetime.strptime(f"{current_date} {clean_time_in}", "%Y-%m-%d %H:%M:%S")
                    
                    elapsed_seconds = (now - time_in_obj).total_seconds()
                    if elapsed_seconds > 300:  # 5 minutes active anti-duplicate check
                        hours = int(elapsed_seconds // 3600)
                        minutes = int((elapsed_seconds % 3600) // 60)
                        calculated_duration = f"{hours:02d}:{minutes:02d}"

                        cursor.execute('''
                            UPDATE Attendance SET TimeOut = ?, TotalDutyHours = ?, Confidence = ? WHERE ID = ?
                        ''', (timestamp_with_date, calculated_duration, float(confidence), record_id))
                        conn.commit()

                conn.close()
                self.root.after(0, self.refresh_logs)
            except Exception as e:
                print(f"Logging worker exception caught: {e}")

    def refresh_logs(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            query = '''
                SELECT a.ID, a.EmployeeID, e.Name, e.FatherName, e.Department, e.Designation, a.TimeIn, a.TimeOut, a.TotalDutyHours
                FROM Attendance a LEFT JOIN Employees e ON a.EmployeeID = e.EmployeeID ORDER BY a.ID DESC LIMIT 50
            '''
            cursor.execute(query)
            rows = cursor.fetchall()
            for index, row in enumerate(rows):
                self.tree.insert("", tk.END, values=(index + 1, row[1], row[2], row[3], row[4], row[5], row[6], row[7] if row[7] != "" else "---", row[8]))
        except sqlite3.OperationalError:
            pass
        conn.close()

if __name__ == "__main__":
    root_window = tk.Tk()
    app_instance = HighEfficiencyCCTVEngine(root_window)
    root_window.mainloop()