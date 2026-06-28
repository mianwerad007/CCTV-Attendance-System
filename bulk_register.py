import os
import cv2
import sqlite3
import numpy as np
from insightface.app import FaceAnalysis
from db_helper import init_db, DB_PATH

def bulk_train_system(dataset_root_dir):
    # Ensure database tables exist cleanly
    init_db()
    
    # Initialize Core InsightFace Face Analysis Pipeline
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    if not os.path.exists(dataset_root_dir):
        print(f"Creating empty folder path structure: '{dataset_root_dir}'")
        os.makedirs(dataset_root_dir)
        print(f"Please drop your employee picture folders inside '{dataset_root_dir}/' and restart.")
        return

    # Scan the folder for Employee IDs
    employee_folders = [f for f in os.listdir(dataset_root_dir) if os.path.isdir(os.path.join(dataset_root_dir, f))]
    
    if not employee_folders:
        print(f"No employee folders found inside '{dataset_root_dir}/'. Ensure folders match Employee IDs.")
        return

    print(f"\nStarting training data compilation matrix for {len(employee_folders)} profiles...\n" + "-"*60)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for emp_id in employee_folders:
        emp_folder_path = os.path.join(dataset_root_dir, emp_id)
        
        print(f"\n[PROFILE SETUP] Configuring Profile ID: {emp_id}")
        emp_name = input(f"  Enter Full Name for {emp_id}: ").strip()
        emp_father = input(f"  Enter Father's Name for {emp_id}: ").strip()
        emp_dept = input(f"  Enter Department for {emp_id}: ").strip()
        emp_desig = input(f"  Enter Designation for {emp_id}: ").strip()
        
        image_embeddings = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        # Parse picture dataset files for this employee folder
        for img_name in os.listdir(emp_folder_path):
            if not img_name.lower().endswith(valid_extensions):
                continue
                
            img_path = os.path.join(emp_folder_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            faces = app.get(img)
            if len(faces) == 0:
                print(f"  [Warning] No visible face detected in image file: {img_name}")
                continue
                
            # Extract normalized vector features
            face_embedding = faces[0].normed_embedding
            image_embeddings.append(face_embedding)
            print(f"  [Feature OK] Processed file: {img_name}")

        if image_embeddings:
            # Generate the mathematical average embedding from images dataset
            averaged_embedding = np.mean(image_embeddings, axis=0)
            averaged_embedding /= np.linalg.norm(averaged_embedding)
            
            # Save raw bytes representation
            blob_data = averaged_embedding.tobytes()
            
            cursor.execute('''
                INSERT OR REPLACE INTO Employees (EmployeeID, Name, FatherName, Department, Designation, FaceEncoding, Status)
                VALUES (?, ?, ?, ?, ?, ?, 'Active')
            ''', (emp_id, emp_name, emp_father, emp_dept, emp_desig, blob_data))
            
            conn.commit()
            print(f"==> Profile record successfully updated for: {emp_name}")
        else:
            print(f"==> [ERROR] Training aborted for {emp_id}: No valid faces found.")

    conn.close()
    print("\n" + "-"*60 + "\nAll employee photo datasets successfully compiled and saved to SQLite!")

if __name__ == "__main__":
    bulk_train_system("employee_faces")