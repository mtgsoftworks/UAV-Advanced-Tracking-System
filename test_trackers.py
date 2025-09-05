import sys
import os
sys.path.append('.')
from uav_tracking_opencv_trackers import OpenCVTrackerUAV

def test_tracker(tracker_type, duration_seconds=30):
    """Belirtilen tracker'ı test et"""
    print(f"\n🧪 {tracker_type} Tracker Test Başlıyor...")
    print("=" * 50)
    
    try:
        # Tracker'ı oluştur ve test et
        uav_tracker = OpenCVTrackerUAV(
            video_path="Sahneler/sahne1.mp4",
            model_path="best.pt",
            tracker_type=tracker_type
        )
        
        # Kısa süre test et
        frame_count = 0
        max_frames = duration_seconds * 25  # ~25 FPS varsayımı
        
        while frame_count < max_frames:
            ret, frame = uav_tracker.cap.read()
            if not ret:
                print("📹 Video bitti")
                break
                
            uav_tracker.frame_count += 1
            frame_count += 1
            
            if uav_tracker.tracking:
                # Tracking modu
                if not uav_tracker._update_tracking(frame):
                    uav_tracker._reset_tracking()
            else:
                # Tarama modu
                if uav_tracker.roi_scan_delay <= 0:
                    bbox, confidence = uav_tracker._detect_uav_in_roi(frame, uav_tracker.current_roi_index)
                    
                    if bbox:
                        uav_tracker.stability_frames += 1
                        if uav_tracker.stability_frames >= uav_tracker.STABILITY_THRESHOLD:
                            if uav_tracker._initialize_tracking(frame, bbox):
                                uav_tracker.detection_count += 1
                            else:
                                uav_tracker.stability_frames = 0
                    else:
                        uav_tracker.stability_frames = 0
                        uav_tracker.current_roi_index = (uav_tracker.current_roi_index + 1) % len(uav_tracker.roi_grid)
                        uav_tracker.roi_scan_delay = uav_tracker.ROI_DELAY_FRAMES
                else:
                    uav_tracker.roi_scan_delay -= 1
            
            # Her 100 frame'de durum raporu
            if frame_count % 100 == 0:
                status = "TRACKING" if uav_tracker.tracking else "SCANNING"
                print(f"⏱️  Frame {frame_count}/{max_frames} - Durum: {status} - Tespit: {uav_tracker.detection_count}")
        
        # Test sonuçları
        print(f"\n✅ {tracker_type} Test Tamamlandı!")
        print(f"   İşlenen Frame: {frame_count}")
        print(f"   Tespit Sayısı: {uav_tracker.detection_count}")
        print(f"   Tracking Durumu: {'AKTİF' if uav_tracker.tracking else 'PASİF'}")
        
        # Temizlik
        uav_tracker.cap.release()
        
        return {
            'tracker': tracker_type,
            'frames': frame_count,
            'detections': uav_tracker.detection_count,
            'tracking_active': uav_tracker.tracking
        }
        
    except Exception as e:
        print(f"❌ {tracker_type} Test Hatası: {e}")
        return {'tracker': tracker_type, 'error': str(e)}

def main():
    """Tüm tracker'ları test et"""
    trackers = ["CSRT", "KCF", "MOSSE", "MIL"]
    results = []
    
    print("🎯 OpenCV Tracker'ları Test Ediliyor...")
    print("⏱️  Her tracker 30 saniye test edilecek\n")
    
    for tracker in trackers:
        try:
            result = test_tracker(tracker, duration_seconds=30)
            results.append(result)
        except KeyboardInterrupt:
            print(f"\n🛑 Test {tracker} için durduruldu")
            break
        except Exception as e:
            print(f"❌ {tracker} genel hatası: {e}")
            results.append({'tracker': tracker, 'error': str(e)})
    
    # Sonuçları özetle
    print("\n" + "="*60)
    print("📊 TEST SONUÇLARI")
    print("="*60)
    
    for result in results:
        if 'error' in result:
            print(f"❌ {result['tracker']}: HATA - {result['error']}")
        else:
            print(f"✅ {result['tracker']}:")
            print(f"   Frame: {result['frames']}")
            print(f"   Tespit: {result['detections']}")
            print(f"   Tracking: {result['tracking_active']}")
            print()

if __name__ == "__main__":
    main() 