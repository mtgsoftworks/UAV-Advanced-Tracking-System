import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Qt/Wayland hatası için önlem

import cv2
import time
import torch
from ultralytics import YOLO
import numpy as np
import math
from collections import deque

# --- TRACKER SINIFI ---
class OpenCVTrackerUAV:
    def __init__(self, video_path="Sahneler/sahne1.mp4", model_path="weights/yolov8s-E150.pt", tracker_type="CSRT"):
        """
        OpenCV Tracker'ları kullanarak UAV takibi
        tracker_type: "CSRT", "KCF", "MOSSE", "MIL", "BOOSTING", "MEDIANFLOW", "TLD"
        """
        self.video_path = video_path
        self.model_path = model_path
        self.tracker_type = tracker_type
        
        # Video ve model yükleme
        self.cap = cv2.VideoCapture(video_path)
        self.device = self._select_best_device()
        self.model = self._load_model_safely(model_path)
        
        # Video bilgileri
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"🎬 Video: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        print(f"🖥️  Seçilen Cihaz: {self.device.upper()}")
        print(f"📡 OpenCV Tracker: {tracker_type}")
        
        if self.device == "cuda":
            print(f"🚀 GPU Modu Aktif - CUDA {torch.version.cuda}")
            print(f"💾 GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print(f"🖥️ CPU Modu Aktif")
        
        # Tracker durumu
        self.tracker = None
        self.tracking = False
        self.uav_bbox = None
        
        # ROI sistemi (4 büyük bölge)
        self.roi_grid = self._create_roi_grid()
        self.current_roi_index = 0
        self.roi_scan_delay = 0
        self.ROI_DELAY_FRAMES = 15
        
        # UAV sınıf isimleri (custom model için genişletildi)
        self.UAV_CLASSES = {
            'iha', 'uav', 'drone', 'aircraft', 'airplane', 'plane', 
            'quadcopter', 'helicopter', 'vehicle', 'flying',
            'datasetv2', 'dataset', 'v3', '2024'  # Custom model class
        }
        
        # Detection parametreleri (Akademik araştırma bulgularına göre optimize)
        self.CONF_THRESHOLD = 0.2   # Ana tespit için optimal threshold (%20)
        self.TRACK_CONF_THRESHOLD = 0.15  # Tracking doğrulama için daha düşük threshold (%15)
        self.IOU_THRESHOLD = 0.4    # NMS threshold (daha iyi NMS performansı)
        self.MAX_DET = 100          # Maximum detections
        
        print(f"📊 OpenCV Tracker Optimized Thresholds:")
        print(f"   • Ana Tespit: {self.CONF_THRESHOLD*100:.0f}%")
        print(f"   • Tracking Verify: {self.TRACK_CONF_THRESHOLD*100:.0f}%") 
        print(f"   • IOU (NMS): {self.IOU_THRESHOLD}")
        print(f"   • Tracker Type: {tracker_type}")
        
        # Performans takibi
        self.frame_count = 0
        self.track_frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
        # Stabilite için gecikme
        self.stability_frames = 0
        self.STABILITY_THRESHOLD = 5

    def _select_best_device(self):
        """
        GPU'yu önce deneyin, sorun varsa CPU'ya geçin.
        """
        print("🔍 OpenCV Tracker - Cihaz seçimi yapılıyor...")
        
        # Önce CUDA varlığını kontrol et
        if not torch.cuda.is_available():
            print("❌ CUDA mevcut değil. CPU kullanılacak.")
            return "cpu"
        
        try:
            # GPU'yu test et
            print("🚀 GPU test ediliyor...")
            torch.cuda.empty_cache()  # GPU belleğini temizle
            
            # Basit bir GPU testi yap
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            
            print("✅ GPU test başarılı! GPU kullanılacak.")
            return "cuda"
            
        except Exception as e:
            print(f"❌ GPU testi başarısız: {e}")
            print("🔄 CPU'ya geçiliyor...")
            return "cpu"

    def _load_model_safely(self, model_path):
        """
        Modeli güvenli şekilde yükler ve device'a atar.
        """
        print(f"📡 OpenCV Tracker - Model yükleniyor: {model_path}")
        
        for attempt in range(3):  # 3 deneme
            try:
                print(f"🔄 Deneme {attempt + 1}/3...")
                
                # Model yükle
                model = YOLO(model_path)
                
                # Model'i device'a ata
                model.to(self.device)
                
                # Küçük bir test yap
                test_results = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), 
                                           verbose=False, device=self.device)
                
                print(f"✅ Model başarıyla yüklendi ve test edildi ({self.device.upper()})!")
                return model
                
            except Exception as e:
                print(f"❌ Model yükleme hatası (Deneme {attempt + 1}/3): {e}")
                
                # GPU hatası ise CPU'ya geç
                if self.device == "cuda" and attempt < 2:
                    print("🔄 GPU hatası nedeniyle CPU'ya geçiliyor...")
                    self.device = "cpu"
                    torch.cuda.empty_cache()
                
                if attempt < 2:
                    print("⏳ 2 saniye bekleniyor...")
                    time.sleep(2)
        
        raise Exception(f"❌ Model {model_path} yüklenemedi! 3 deneme başarısız.")

    def _create_roi_grid(self):
        """4 büyük ROI bölgesi oluştur (2x2 grid)"""
        roi_width = self.frame_width // 2
        roi_height = self.frame_height // 2
        overlap = 100  # Overlap ekle
        
        rois = [
            (0, 0, roi_width + overlap, roi_height + overlap),  # Sol üst
            (roi_width - overlap, 0, roi_width + overlap, roi_height + overlap),  # Sağ üst
            (0, roi_height - overlap, roi_width + overlap, roi_height + overlap),  # Sol alt
            (roi_width - overlap, roi_height - overlap, roi_width + overlap, roi_height + overlap)  # Sağ alt
        ]
        return rois

    def _create_tracker(self):
        """Belirtilen tipte tracker oluştur (LearnOpenCV best practices)"""
        # Tracker öncelik sırası (performans ve doğruluk açısından)
        tracker_priority = [
            self.tracker_type,  # Kullanıcının seçtiği
            "CSRT",             # En doğru ama yavaş
            "KCF",              # İyi performans dengesi
            "MOSSE",            # En hızlı
            "MIL"               # Temel tracker
        ]
        
        for tracker_name in tracker_priority:
            try:
                print(f"🔄 {tracker_name} tracker deneniyor...")
                
                if tracker_name == "CSRT":
                    # En doğru tracker - channel-spatial reliability
                    try:
                        self.tracker = cv2.TrackerCSRT_create()
                    except:
                        self.tracker = cv2.legacy.TrackerCSRT_create()
                        
                elif tracker_name == "KCF":
                    # Hız-doğruluk dengesi iyi
                    try:
                        self.tracker = cv2.TrackerKCF_create()
                    except:
                        self.tracker = cv2.legacy.TrackerKCF_create()
                        
                elif tracker_name == "MOSSE":
                    # En hızlı tracker (450+ FPS)
                    try:
                        self.tracker = cv2.TrackerMOSSE_create()
                    except:
                        try:
                            self.tracker = cv2.legacy.TrackerMOSSE_create()
                        except:
                            print(f"⚠️ MOSSE tracker mevcut değil, atlanıyor...")
                            continue
                            
                elif tracker_name == "MIL":
                    # Multiple Instance Learning - temel tracker
                    try:
                        self.tracker = cv2.TrackerMIL_create()
                    except:
                        try:
                            self.tracker = cv2.legacy.TrackerMIL_create()
                        except:
                            print(f"⚠️ MIL tracker mevcut değil, atlanıyor...")
                            continue
                            
                elif tracker_name == "BOOSTING":
                    try:
                        self.tracker = cv2.TrackerBoosting_create()
                    except:
                        try:
                            self.tracker = cv2.legacy.TrackerBoosting_create()
                        except:
                            print(f"⚠️ BOOSTING tracker mevcut değil, atlanıyor...")
                            continue
                            
                elif tracker_name == "MEDIANFLOW":
                    try:
                        self.tracker = cv2.TrackerMedianFlow_create()
                    except:
                        try:
                            self.tracker = cv2.legacy.TrackerMedianFlow_create()
                        except:
                            print(f"⚠️ MEDIANFLOW tracker mevcut değil, atlanıyor...")
                            continue
                            
                elif tracker_name == "TLD":
                    try:
                        self.tracker = cv2.TrackerTLD_create()
                    except:
                        try:
                            self.tracker = cv2.legacy.TrackerTLD_create()
                        except:
                            print(f"⚠️ TLD tracker mevcut değil, atlanıyor...")
                            continue
                else:
                    continue
                
                self.tracker_type = tracker_name
                print(f"✅ {tracker_name} tracker başarıyla oluşturuldu!")
                return True
                
            except Exception as e:
                print(f"❌ {tracker_name} tracker hatası: {e}")
                continue
        
        print(f"❌ Hiçbir tracker oluşturulamadı!")
        return False

    def _detect_uav_in_roi(self, frame, roi_index):
        """ROI içinde UAV tespiti yap"""
        x, y, w, h = self.roi_grid[roi_index]
        x, y = max(0, x), max(0, y)
        w = min(w, self.frame_width - x)
        h = min(h, self.frame_height - y)
        
        roi_frame = frame[y:y+h, x:x+w]
        
        results = self.model(
            roi_frame,
            conf=self.CONF_THRESHOLD,  # Optimize edilmiş threshold
            iou=self.IOU_THRESHOLD,    # Optimize edilmiş IOU
            verbose=False,
            device=self.device
        )
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # Class kontrolü
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id].lower()
                confidence = float(box.conf[0])
                
                # UAV sınıfı kontrolü
                is_uav = any(uav_class in class_name for uav_class in self.UAV_CLASSES)
                
                if is_uav and confidence > self.CONF_THRESHOLD:
                    # Bbox koordinatlarını orijinal frame'e çevir
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    global_x1, global_y1 = x + x1, y + y1
                    global_x2, global_y2 = x + x2, y + y2
                    
                    bbox_w, bbox_h = global_x2 - global_x1, global_y2 - global_y1
                    
                    # Minimum boyut kontrolü (optimized: %2.5 of frame)
                    min_w = int(self.frame_width * 0.025)   # Akademik öneriler
                    min_h = int(self.frame_height * 0.025)
                    if bbox_w > min_w and bbox_h > min_h:
                        print(f"✅ UAV tespit edildi! Class: {class_name}, Conf: {confidence:.2f}, ROI: {roi_index+1}")
                        return (global_x1, global_y1, bbox_w, bbox_h), confidence
        
        return None, 0.0

    def _initialize_tracking(self, frame, bbox):
        """Tracker'ı başlat"""
        if not self._create_tracker():
            return False
            
        # Bbox'ı biraz genişlet (optimized padding: %15)
        x, y, w, h = bbox
        padding_ratio = 0.15  # LearnOpenCV optimal padding
        padding_w = int(w * padding_ratio)
        padding_h = int(h * padding_ratio)
        
        x = max(0, x - padding_w)
        y = max(0, y - padding_h)
        w = min(w + 2*padding_w, self.frame_width - x)
        h = min(h + 2*padding_h, self.frame_height - y)
        
        padded_bbox = (x, y, w, h)
        
        # Tracker'ı initialize et
        success = self.tracker.init(frame, padded_bbox)
        if success:
            self.tracking = True
            self.uav_bbox = padded_bbox
            self.track_frame_count = 0
            print(f"🎯 {self.tracker_type} Tracker başlatıldı!")
            return True
        else:
            print(f"❌ {self.tracker_type} Tracker başlatılamadı!")
            return False

    def _update_tracking(self, frame):
        """Tracker güncelle"""
        success, bbox = self.tracker.update(frame)
        
        if success:
            x, y, w, h = map(int, bbox)
            
            # Sınır kontrolü
            x = max(0, min(x, self.frame_width - w))
            y = max(0, min(y, self.frame_height - h))
            w = min(w, self.frame_width - x)
            h = min(h, self.frame_height - y)
            
            self.uav_bbox = (x, y, w, h)
            self.track_frame_count += 1
            
            # Tracker doğrulama (her 60 frame'de bir)
            if self.track_frame_count % 60 == 0:
                if not self._verify_tracking(frame):
                    return False
            
            return True
        else:
            print(f"⚠️ {self.tracker_type} Tracker nesneyi kaybetti")
            return False

    def _verify_tracking(self, frame):
        """Tracking'i YOLO ile doğrula"""
        if not self.uav_bbox:
            return False
            
        x, y, w, h = self.uav_bbox
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return False
            
        results = self.model(roi, conf=self.TRACK_CONF_THRESHOLD, verbose=False, device=self.device)
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id].lower()
                is_uav = any(uav_class in class_name for uav_class in self.UAV_CLASSES)
                
                if is_uav:
                    return True
        
        print("⚠️ Tracking doğrulanamadı, tarama moduna geçiliyor")
        return False

    def _reset_tracking(self):
        """Tracking'i sıfırla"""
        self.tracking = False
        self.tracker = None
        self.uav_bbox = None
        self.current_roi_index = 0
        self.roi_scan_delay = 0
        self.stability_frames = 0

    def _draw_interface(self, frame):
        """Arayüz elemanlarını çiz"""
        # Tracking durumu
        if self.tracking and self.uav_bbox:
            x, y, w, h = self.uav_bbox
            # UAV kutusunu çiz
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, f"{self.tracker_type} TRACKER", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {self.track_frame_count}", (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # ROI'ları çiz
            for i, (rx, ry, rw, rh) in enumerate(self.roi_grid):
                color = (0, 255, 255) if i == self.current_roi_index else (100, 100, 100)
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
                cv2.putText(frame, f"ROI{i+1}", (rx + 10, ry + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Durum bilgileri
        status_text = f"{self.tracker_type} TAKIP" if self.tracking else f"TARAMA (ROI {self.current_roi_index + 1}/4)"
        status_color = (0, 255, 0) if self.tracking else (0, 255, 255)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Performans bilgileri
        fps = self.frame_count / (time.time() - self.start_time) if time.time() - self.start_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, self.frame_height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, self.frame_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        """Ana çalışma döngüsü"""
        print(f"🚀 {self.tracker_type} Tracker ile UAV takibi başlatılıyor...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("📹 Video bitti")
                break
                
            self.frame_count += 1
            
            if self.tracking:
                # Tracking modu
                if not self._update_tracking(frame):
                    self._reset_tracking()
            else:
                # Tarama modu
                if self.roi_scan_delay <= 0:
                    bbox, confidence = self._detect_uav_in_roi(frame, self.current_roi_index)
                    
                    if bbox:
                        self.stability_frames += 1
                        if self.stability_frames >= self.STABILITY_THRESHOLD:
                            if self._initialize_tracking(frame, bbox):
                                self.detection_count += 1
                            else:
                                self.stability_frames = 0
                    else:
                        self.stability_frames = 0
                        # Sonraki ROI'ye geç
                        self.current_roi_index = (self.current_roi_index + 1) % len(self.roi_grid)
                        self.roi_scan_delay = self.ROI_DELAY_FRAMES
                else:
                    self.roi_scan_delay -= 1
            
            # Arayüzü çiz
            self._draw_interface(frame)
            
            # Görüntüyü göster
            cv2.imshow(f"{self.tracker_type} UAV Tracker", frame)
            
            # Çıkış kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset
                self._reset_tracking()
                print("🔄 Tracking sıfırlandı")
        
        # Temizlik
        self.cap.release()
        cv2.destroyAllWindows()
        
        # İstatistikler
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time
        print(f"\n📊 Sonuçlar:")
        print(f"   Tracker: {self.tracker_type}")
        print(f"   İşlenen Frame: {self.frame_count}")
        print(f"   Tespit Sayısı: {self.detection_count}")
        print(f"   Ortalama FPS: {avg_fps:.2f}")
        print(f"   Toplam Süre: {total_time:.2f}s")

def main():
    """Ana fonksiyon - Farklı tracker'ları test et"""
    # LearnOpenCV'ye göre performans sıralaması
    trackers = [
        ("CSRT", "🎯 En Doğru (Yavaş) - UAV tracking için önerilen"),
        ("KCF", "⚡ Dengeli Performans - Hız/Doğruluk dengesi"),  
        ("MOSSE", "🚀 En Hızlı (450 FPS) - Basit tracking"),
        ("MIL", "📚 Temel Tracker - Multiple Instance Learning"),
        ("BOOSTING", "🔧 İlk Tracker - Temel boosting"),
        ("MEDIANFLOW", "📊 Tahmin Edilebilir Hareket"),
        ("TLD", "🔍 Oklüzyon Altında İyi")
    ]
    
    print("🎯 Mevcut OpenCV Tracker'lar (LearnOpenCV Önerileri):")
    for i, (tracker_name, description) in enumerate(trackers):
        print(f"   {i+1}. {tracker_name:<12} - {description}")
    
    print(f"\n💡 UAV Tracking için önerilen sıralama:")
    print(f"   1. CSRT (En doğru ama yavaş)")
    print(f"   2. KCF  (İyi denge)")
    print(f"   3. MOSSE (En hızlı)")
    
    print(f"\n📋 Detaylı Tracker Bilgileri:")
    print(f"═" * 80)
    
    print(f"🎯 1. CSRT (Discriminative Correlation Filter):")
    print(f"   • Doğruluk: ⭐⭐⭐⭐⭐ (En yüksek)")
    print(f"   • Hız: ⭐⭐ (25 FPS)")
    print(f"   • Avantaj: Non-rectangular nesneleri mükemmel takip eder")
    print(f"   • Dezavantaj: Yavaş, hesaplama yoğun")
    print(f"   • UAV için: 🟢 EN ÖNERİLEN (hassas tracking)")
    
    print(f"\n⚡ 2. KCF (Kernelized Correlation Filters):")
    print(f"   • Doğruluk: ⭐⭐⭐⭐")
    print(f"   • Hız: ⭐⭐⭐⭐")
    print(f"   • Avantaj: Mükemmel hız/doğruluk dengesi")
    print(f"   • Dezavantaj: Tam oklüzyon altında zayıf")
    print(f"   • UAV için: 🟢 İYİ SEÇİM (genel amaçlı)")
    
    print(f"\n🚀 3. MOSSE (Minimum Output Sum of Squared Error):")
    print(f"   • Doğruluk: ⭐⭐⭐")
    print(f"   • Hız: ⭐⭐⭐⭐⭐ (450+ FPS!)")
    print(f"   • Avantaj: Ultra hızlı, lighting değişimlerine dayanıklı")
    print(f"   • Dezavantaj: Daha az hassas")
    print(f"   • UAV için: 🟡 HIZ GEREKİYORSA (real-time)")
    
    print(f"\n📚 4. MIL (Multiple Instance Learning):")
    print(f"   • Doğruluk: ⭐⭐⭐")
    print(f"   • Hız: ⭐⭐⭐")
    print(f"   • Avantaj: Kısmi oklüzyon altında iyi")
    print(f"   • Dezavantaj: Tracking failure raporlama zayıf")
    print(f"   • UAV için: 🟡 TEMEL (basit senaryolar)")
    
    print(f"\n🔧 5. BOOSTING:")
    print(f"   • Doğruluk: ⭐⭐")
    print(f"   • Hız: ⭐⭐")
    print(f"   • Avantaj: Basit implementasyon")
    print(f"   • Dezavantaj: Drift problemi, eski teknoloji")
    print(f"   • UAV için: 🔴 ÖNERİLMEZ (outdated)")
    
    print(f"\n📊 6. MEDIANFLOW:")
    print(f"   • Doğruluk: ⭐⭐⭐")
    print(f"   • Hız: ⭐⭐⭐")
    print(f"   • Avantaj: Mükemmel failure detection")
    print(f"   • Dezavantaj: Hızlı hareket altında başarısız")
    print(f"   • UAV için: 🔴 UYGUN DEĞİL (slow motion only)")
    
    print(f"\n🔍 7. TLD (Tracking-Learning-Detection):")
    print(f"   • Doğruluk: ⭐⭐⭐⭐")
    print(f"   • Hız: ⭐⭐")
    print(f"   • Avantaj: Oklüzyon altında en iyi")
    print(f"   • Dezavantaj: Çok fazla false positive")
    print(f"   • UAV için: 🟡 OKLUZYONLU ORTAMLARDA (experimental)")
    
    print(f"\n🏆 LearnOpenCV Tavsiyesi: UAV tracking için CSRT (1) > KCF (2) > MOSSE (3)")
    print(f"═" * 80)
    
    while True:
        try:
            choice = input(f"\nTracker seçin (1-{len(trackers)}) veya 'q' çıkış: ").strip()
            if choice.lower() == 'q':
                break
                
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(trackers):
                selected_tracker = trackers[choice_idx][0]
                print(f"\n🚀 {selected_tracker} tracker başlatılıyor...")
                print(f"📋 {trackers[choice_idx][1]}")
                
                # Tracker'ı başlat
                uav_tracker = OpenCVTrackerUAV(
                    video_path="Sahneler/sahne1.mp4",
                    model_path="weights/yolov8s-E150.pt",
                    tracker_type=selected_tracker
                )
                uav_tracker.run()
            else:
                print("❌ Geçersiz seçim!")
        except ValueError:
            print("❌ Lütfen geçerli bir sayı girin!")
        except KeyboardInterrupt:
            print("\n🛑 Program sonlandırıldı")
            break

if __name__ == "__main__":
    main() 