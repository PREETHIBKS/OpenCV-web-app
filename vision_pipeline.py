import cv2
import numpy as np
import time
import os
from collections import deque

class VisionPipeline:
    def __init__(self):
        # Load Haar Cascades using built-in OpenCV data path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Motion detection state
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)
        self.motion_threshold = 1500
        
        # Color tracking state
        # Default: Track vibrant blue/cyan
        self.hsv_lower = np.array([90, 80, 50])
        self.hsv_upper = np.array([130, 255, 255])
        self.tracking_trail = deque(maxlen=32)
        self.target_bgr = (255, 210, 0) # Default cyan overlay
        
        # Face recognition templates database
        # Structure: { "name": [gray_face_template1, gray_face_template2, ...] }
        self.registered_faces = {}
        self.recognition_threshold = 0.60
        
        # Load any previously saved face templates
        self.templates_dir = os.path.join(os.path.dirname(__file__), 'face_templates')
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir)
        self.load_registered_faces()

    def load_registered_faces(self):
        """Loads face templates from the templates directory."""
        try:
            for filename in os.listdir(self.templates_dir):
                if filename.endswith('.png'):
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        name = parts[0]
                        img_path = os.path.join(self.templates_dir, filename)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img_resized = cv2.resize(img, (100, 100))
                            if name not in self.registered_faces:
                                self.registered_faces[name] = []
                            self.registered_faces[name].append(img_resized)
            print(f"Loaded templates for faces: {list(self.registered_faces.keys())}")
        except Exception as e:
            print(f"Error loading face templates: {e}")

    def register_face(self, frame, name):
        """Detects a face in the current frame and registers it under the provided name."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            # Take the largest face found
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (100, 100))
            
            # Save template to disk
            timestamp = int(time.time())
            filename = f"{name}_{timestamp}.png"
            filepath = os.path.join(self.templates_dir, filename)
            cv2.imwrite(filepath, face_resized)
            
            # Add to memory cache
            if name not in self.registered_faces:
                self.registered_faces[name] = []
            self.registered_faces[name].append(face_resized)
            return True, f"Successfully registered face for '{name}'."
        
        # Fallback: Crop the center 45% region of the frame as face signature
        h_img, w_img = gray.shape
        w_crop = int(w_img * 0.45)
        h_crop = int(h_img * 0.45)
        x = (w_img - w_crop) // 2
        y = (h_img - h_crop) // 2
        face_roi = gray[y:y+h_crop, x:x+w_crop]
        face_resized = cv2.resize(face_roi, (100, 100))
        
        # Save template to disk
        timestamp = int(time.time())
        filename = f"{name}_{timestamp}.png"
        filepath = os.path.join(self.templates_dir, filename)
        cv2.imwrite(filepath, face_resized)
        
        # Add to memory cache
        if name not in self.registered_faces:
            self.registered_faces[name] = []
        self.registered_faces[name].append(face_resized)
        return True, f"Registered using center-frame capture fallback for '{name}'."

    def draw_futuristic_hud(self, frame, title="HUD ACTIVE", fps=0):
        """Helper to draw a high-tech glowing sci-fi HUD frame."""
        h, w, _ = frame.shape
        # Tech borders/corners
        color = (255, 210, 0) # Glowing Cyber Cyan/Blue in BGR: Cyan is (255, 210, 0)
        thickness = 2
        length = 25
        
        # Top-left corner
        cv2.line(frame, (20, 20), (20 + length, 20), color, thickness)
        cv2.line(frame, (20, 20), (20, 20 + length), color, thickness)
        # Top-right corner
        cv2.line(frame, (w - 20, 20), (w - 20 - length, 20), color, thickness)
        cv2.line(frame, (w - 20, 20), (w - 20, 20 + length), color, thickness)
        # Bottom-left corner
        cv2.line(frame, (20, h - 20), (20 + length, h - 20), color, thickness)
        cv2.line(frame, (20, h - 20), (20, h - 20 - length), color, thickness)
        # Bottom-right corner
        cv2.line(frame, (w - 20, h - 20), (w - 20 - length, h - 20), color, thickness)
        cv2.line(frame, (w - 20, h - 20), (w - 20, h - 20 - length), color, thickness)
        
        # Subdued tech lines
        cv2.line(frame, (20, 30), (w - 20, 30), (120, 100, 0), 1)
        cv2.line(frame, (20, h - 30), (w - 20, h - 30), (120, 100, 0), 1)

        # Header Title
        cv2.putText(frame, f"// SYS_MODE: {title}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        # FPS display
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1, cv2.LINE_AA)

    def process_motion_detection(self, frame, fps=0):
        """1. Moving Object Detection: adaptive background subtraction and contours."""
        # Clean frame with blur to reduce high-frequency noise
        blurred = cv2.GaussianBlur(frame, (15, 15), 0)
        fg_mask = self.background_subtractor.apply(blurred)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        # Find contours of moving items
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_count = 0
        
        for contour in contours:
            # Filter small changes to avoid wind/noise detection
            if cv2.contourArea(contour) < self.motion_threshold:
                continue
            
            motion_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            
            # Glowing neon bounding box (Green)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw glowing status dots at corners
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            cv2.circle(frame, (x+w, y+h), 4, (0, 255, 0), -1)
            
            # Add target label
            cv2.putText(frame, f"OBJ_{motion_count:02d}", (x, y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            
        # Draw tech GUI overlays
        self.draw_futuristic_hud(frame, "MOTION_DETECTION_ACTIVE", fps)
        
        # Status footer
        status_color = (0, 255, 0) if motion_count > 0 else (0, 100, 255)
        status_text = f"MOTION LEVEL: {'DETECTED' if motion_count > 0 else 'STANDBY'} ({motion_count} active vectors)"
        cv2.putText(frame, status_text, (30, frame.shape[0] - 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
        
        return frame

    def process_face_detection(self, frame, fps=0):
        """2. Face & Eye Detection: detects facial geometry with detailed HUD overlays."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Fast multi-scale detection
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        
        for idx, (x, y, w, h) in enumerate(faces):
            # 1. Tech styling face bounding bracket (Neon Cyan)
            cyan = (255, 255, 0)
            thickness = 2
            r = 15 # corner length
            # Draw custom corner brackets for a sci-fi target scanner look
            cv2.line(frame, (x, y), (x + r, y), cyan, thickness)
            cv2.line(frame, (x, y), (x, y + r), cyan, thickness)
            
            cv2.line(frame, (x + w, y), (x + w - r, y), cyan, thickness)
            cv2.line(frame, (x + w, y), (x + w, y + r), cyan, thickness)
            
            cv2.line(frame, (x, y + h), (x + r, y + h), cyan, thickness)
            cv2.line(frame, (x, y + h), (x, y + h - r), cyan, thickness)
            
            cv2.line(frame, (x + w, y + h), (x + w - r, y + h), cyan, thickness)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - r), cyan, thickness)
            
            # Subtle face box boundary
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 1, lineType=cv2.LINE_8)
            
            # Face metadata overlay
            label = f"SUBJECT_{idx+1:02d}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cyan, 1, cv2.LINE_AA)
            
            # Crop region of interest for eyes detection (restricting search area to upper 55% of face to avoid false positives like nostrils/mouth)
            face_roi_gray_upper = gray[y:y+int(h*0.55), x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes within the upper face ROI
            eyes = self.eye_cascade.detectMultiScale(face_roi_gray_upper, 1.15, 4, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes:
                # Draw circular targeting grids on eyes
                cx, cy = ex + ew // 2, ey + eh // 2
                cv2.circle(face_roi_color, (cx, cy), ew // 2, (0, 180, 255), 1, cv2.LINE_AA)
                cv2.circle(face_roi_color, (cx, cy), 3, (0, 100, 255), -1)
                
        # Technical HUD overlays
        self.draw_futuristic_hud(frame, "FACE_SCANNER_ONLINE", fps)
        
        status_text = f"SUBJECTS ACQUIRED: {len(faces)}"
        status_color = (255, 255, 0) if len(faces) > 0 else (0, 150, 255)
        cv2.putText(frame, status_text, (30, frame.shape[0] - 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
        
        return frame

    def process_color_tracking(self, frame, fps=0, custom_hsv=None):
        """3. Color Object Tracking: HSV segmentation with historical trailing paths."""
        if custom_hsv is not None:
            # Expecting dict: { 'lower': [h, s, v], 'upper': [h, s, v] }
            self.hsv_lower = np.array(custom_hsv['lower'])
            self.hsv_upper = np.array(custom_hsv['upper'])
            
        draw_color = self.target_bgr

        # Segment image in HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Clean mask using opening and closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        target_center = None
        if len(contours) > 0:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 800:
                # Get minimum enclosing circle to calculate precise tracking region
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                cx, cy = int(x), int(y)
                target_center = (cx, cy)
                
                # Draw outer glow circle and center tracking crosshairs matching target color
                cv2.circle(frame, (cx, cy), int(radius), draw_color, 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 5, draw_color, -1)
                
                # Crosshairs
                len_ch = 15
                cv2.line(frame, (cx - len_ch, cy), (cx + len_ch, cy), draw_color, 1)
                cv2.line(frame, (cx, cy - len_ch), (cx, cy + len_ch), draw_color, 1)
                
                # Track position overlay
                coord_text = f"TARGET: X:{cx} Y:{cy}"
                cv2.putText(frame, coord_text, (cx + int(radius) + 10, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_color, 1, cv2.LINE_AA)
                
        # Update trailing history path
        self.tracking_trail.appendleft(target_center)
        
        # Render the trailing tracking history with fading line widths matching target color
        for i in range(1, len(self.tracking_trail)):
            if self.tracking_trail[i - 1] is None or self.tracking_trail[i] is None:
                continue
            # Calculate gradient thickness/opacity
            thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
            cv2.line(frame, self.tracking_trail[i - 1], self.tracking_trail[i], draw_color, thickness, cv2.LINE_AA)
            
        # Draw tech GUI overlays
        self.draw_futuristic_hud(frame, "CHROMA_OBJECT_TRACKING", fps)
        
        # Color bar showing target HSV settings
        cv2.rectangle(frame, (30, frame.shape[0] - 65), (150, frame.shape[0] - 50), draw_color, -1)
        cv2.putText(frame, "TARGET RANGE ACTIVE", (160, frame.shape[0] - 53), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        status_text = f"TRACKING STATUS: {'TARGET LOCKED' if target_center else 'SEARCHING OBJECT...'}"
        status_color = draw_color if target_center else (0, 100, 255)
        cv2.putText(frame, status_text, (30, frame.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
        
        return frame

    def process_face_recognition(self, frame, fps=0):
        """4. Face Recognition: high-speed multi-face similarity template matcher with live registration."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            # Crop detected face region
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (100, 100))
            
            best_match_name = "UNKNOWN"
            best_match_score = 0.0
            
            # Compare current face template against registered database templates
            for name, templates in self.registered_faces.items():
                for template in templates:
                    # Normalized correlation matching
                    res = cv2.matchTemplate(face_resized, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > best_match_score:
                        best_match_score = max_val
            
            # Draw HUD visuals
            if best_match_score >= self.recognition_threshold and best_match_name == "UNKNOWN":
                # Find matching user name
                for name, templates in self.registered_faces.items():
                    found = False
                    for template in templates:
                        res = cv2.matchTemplate(face_resized, template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        if abs(max_val - best_match_score) < 0.0001:
                            best_match_name = name
                            found = True
                            break
                    if found:
                        break
            
            # Color schemes based on verification status
            if best_match_name != "UNKNOWN":
                color = (0, 255, 100) # Vibrant green for verified user
                label = f"VERIFIED: {best_match_name.upper()} ({int(best_match_score*100)}%)"
            else:
                color = (0, 200, 255) # Warning yellow for unverified visitor
                label = "VISITOR: UNREGISTERED"
                
            # Draw beautiful bounding details
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.line(frame, (x, y), (x + 15, y), color, 3)
            cv2.line(frame, (x, y), (x, y + 15), color, 3)
            cv2.line(frame, (x + w, y), (x + w - 15, y), color, 3)
            cv2.line(frame, (x + w, y), (x + w, y + 15), color, 3)
            
            # Draw label banner
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            
        # Draw tech GUI overlays
        self.draw_futuristic_hud(frame, "BIOMETRIC_RECOGNITION", fps)
        
        # User base information footer
        db_count = len(self.registered_faces.keys())
        cv2.putText(frame, f"BIOMETRIC ARCHIVE: {db_count} profiles", (30, frame.shape[0] - 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100) if db_count > 0 else (0, 100, 255), 1, cv2.LINE_AA)
        
        return frame

    def process_emotion_detection(self, frame, fps=0):
        """5. Face Emotion Detection: high-speed visual geometric scanner for instant expression feedback."""
        purple = (255, 100, 180)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        
        for (x, y, w, h) in faces:
            # Beautiful facial HUD targeting box (Neon purple)
            cv2.rectangle(frame, (x, y), (x+w, y+h), purple, 1)
            
            # Crop face area for details
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]
            
            # Draw diagnostic layout lines over the face
            # Center axes
            cx, cy = x + w // 2, y + h // 2
            cv2.line(frame, (cx, y), (cx, y + h), (180, 80, 130), 1, cv2.LINE_8)
            cv2.line(frame, (x, cy), (x + w, cy), (180, 80, 130), 1, cv2.LINE_8)
            
            # Detect eyes within the upper face ROI to use as facial anchors (restricting to upper 55% of face to avoid nostrils/mouth)
            face_roi_gray_upper = face_roi_gray[0:int(h*0.55), :]
            eyes = self.eye_cascade.detectMultiScale(face_roi_gray_upper, 1.15, 5, minSize=(15, 15))
            eyes = sorted(eyes, key=lambda e: e[0]) # Sort left to right
            
            # Default fallback parameters
            smile_ratio = 0.0
            eye_ratio = 1.0
            mouth_openness = 0.0
            
            # Analyze smile
            smiles = self.smile_cascade.detectMultiScale(
                face_roi_gray[int(h*0.5):h, :], 1.2, 10, minSize=(25, 15)
            )
            
            smile_detected = len(smiles) > 0
            if smile_detected:
                # Calculate smile width ratio
                sx, sy, sw, sh = smiles[0]
                smile_ratio = float(sw) / w
                
            # Analyze mouth openness via basic image thresholding in lower third of face
            mouth_y_start = int(h * 0.65)
            mouth_y_end = int(h * 0.90)
            mouth_x_start = int(w * 0.25)
            mouth_x_end = int(w * 0.75)
            
            mouth_area = face_roi_gray[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
            frown_detected = False
            if mouth_area.size > 0:
                # Apply OTSU thresholding to find dark open-mouth pixels
                _, mouth_thresh = cv2.threshold(mouth_area, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                mouth_pixels = cv2.countNonZero(mouth_thresh)
                mouth_openness = float(mouth_pixels) / mouth_area.size
                
                # Find contours in the mouth thresholded area to check lip curvature geometry
                m_contours, _ = cv2.findContours(mouth_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(m_contours) > 0:
                    m_contour = max(m_contours, key=cv2.contourArea)
                    mx, my, mw, mh = cv2.boundingRect(m_contour)
                    if mw > 15 and mh > 3:
                        left_pts, mid_pts, right_pts = [], [], []
                        for pt in m_contour:
                            px, py = pt[0][0], pt[0][1]
                            if mx <= px < mx + int(mw * 0.25):
                                left_pts.append(py)
                            elif mx + int(mw * 0.38) <= px < mx + int(mw * 0.62):
                                mid_pts.append(py)
                            elif mx + int(mw * 0.75) <= px <= mx + mw:
                                right_pts.append(py)
                        
                        if len(left_pts) > 0 and len(mid_pts) > 0 and len(right_pts) > 0:
                            corners_y = (sum(left_pts)/len(left_pts) + sum(right_pts)/len(right_pts)) / 2.0
                            middle_y = sum(mid_pts) / len(mid_pts)
                            curvature = corners_y - middle_y
                            
                            # Screen Y increases downwards, so corners_y > middle_y means corners droop down (frown)
                            if curvature > 0.6:
                                frown_detected = True
                            elif curvature < -0.6:
                                smile_ratio = max(smile_ratio, 0.4)
            
            # Classify expression based on visual ratios and mouth curvature
            emotion = "NEUTRAL"
            confidence = 65.0
            
            if smile_detected or smile_ratio > 0.35:
                emotion = "HAPPY"
                confidence = min(85.0 + smile_ratio * 40, 100.0)
            elif frown_detected:
                emotion = "SAD"
                confidence = 82.0
            elif mouth_openness > 0.40:
                emotion = "SURPRISED"
                confidence = min(70.0 + mouth_openness * 60, 100.0)
            elif mouth_openness < 0.15:
                # Test for angry/sad based on eye configurations if eyes are detected
                if len(eyes) >= 2:
                    ex1, ey1, ew1, eh1 = eyes[0]
                    ex2, ey2, ew2, eh2 = eyes[1]
                    eye_dist = abs(ex2 - ex1)
                    if eye_dist < w * 0.42:
                        emotion = "ANGRY"
                        confidence = 78.0
                    else:
                        emotion = "SAD"
                        confidence = 70.0
                else:
                    emotion = "SAD"
                    confidence = 60.0
            else:
                emotion = "NEUTRAL"
                confidence = 75.0
                
            # Draw facial keypoints overlays
            if len(eyes) >= 2:
                # Draw green eye target circles
                for (ex, ey, ew, eh) in eyes[:2]:
                    ecx, ecy = x + ex + ew//2, y + ey + eh//2
                    cv2.circle(frame, (ecx, ecy), 5, (0, 255, 100), -1)
                    cv2.circle(frame, (ecx, ecy), 10, (0, 255, 100), 1)
                    
            # Display current emotion on facial card overlay
            emoji_map = {"HAPPY": "^_^ HAPPY", "SAD": ";_( SAD", "SURPRISED": "o_O SURPRISED", "ANGRY: ": ">_< ANGRY", "NEUTRAL": ".-. NEUTRAL"}
            emo_lbl = emoji_map.get(emotion, emotion)
            
            cv2.putText(frame, f"{emo_lbl} ({confidence:.0f}%)", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, purple, 2, cv2.LINE_AA)
            
            # Interactive analytical metrics overlay next to the face
            # Draw confidence meter
            bar_w = int(w * 0.8)
            bar_x = x + (w - bar_w) // 2
            bar_y = y + h + 15
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 6), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * (confidence / 100.0)), bar_y + 6), purple, -1)
            
        # Draw tech GUI overlays
        self.draw_futuristic_hud(frame, "COGNITIVE_EMOTION_SCAN", fps)
        
        # Bottom diagnostic log display
        active_emo_text = "DIAGNOSTIC STATUS: SCANNING COGNITIVE SIGNALS..."
        if len(faces) > 0:
            active_emo_text = f"DIAGNOSTIC STATUS: FACIAL EXPRESSION QUANTIFIED"
        cv2.putText(frame, active_emo_text, (30, frame.shape[0] - 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, purple, 1, cv2.LINE_AA)
        
        return frame
