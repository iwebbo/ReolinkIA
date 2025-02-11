import cv2
from ultralytics import YOLO

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")  # Remplace par ton modèle si besoin

# Liste des classes à détecter (0 = person, 16 = dog dans COCO)
TARGET_CLASSES = [0, 16]
CONFIDENCE_THRESHOLD = 0.5  # Ajuste cette valeur (0.5 = 50% de confiance)

# URL du flux RTSP (remplace par l'URL de ta caméra)
rtsp_url = "rtsp://USER:PASSWORD@IP.LOCAL:554"

# Ouvrir le flux vidéo
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir le flux RTSP")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le frame")
        break

    # Exécuter YOLOv8 sur l'image
    results = model(frame)

    # Filtrer les détections
    filtered_boxes = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())  # Classe détectée
            confidence = float(box.conf[0].item())  # Confiance de la détection

            if class_id in TARGET_CLASSES and confidence >= CONFIDENCE_THRESHOLD:
                filtered_boxes.append(box)

    # Dessiner uniquement les détections filtrées
    for box in filtered_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Coordonnées bbox
        confidence = float(box.conf[0].item())
        label = f"{model.names[int(box.cls[0].item())]}: {confidence:.2f}"

        # Dessiner la boîte et le texte
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher l'image
    cv2.imshow("YOLOv8 RTSP Detection", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
