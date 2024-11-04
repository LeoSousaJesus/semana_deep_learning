# pip install opencv-python
import cv2
# pip install mediapipe
import mediapipe as mp

# Capturar a câmera
cap = cv2.VideoCapture(0)

# Desenhar os pontos
mp_drawing = mp.solutions.drawing_utils

# Coletar a solução do Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Enquanto a camera estiver aberta
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5) as facemesh:
    while cap.isOpened():
        # sucesso é booleana - 0 e 1
        sucesso,frame = cap.read()
        if not sucesso:
            print('Ignorando o frame vazio da câmera')
            continue
        # transformando de BGR para RGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # processar o frame (OpenCV - MediaPipe)
        saida_facemesh = facemesh.process(frame)
        # transformando de RGB para BGR
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        # vamos desenhar?
        # 1 - Fizemos a detecção do rosto com facemesh.process(frame)
        # 2 - Mostrar essa detecção
        # 3 - for (iteração) - o for é um while compacto
            # 4 - face_landmarks - conjunto das coordenadas
                # (coletamos as coordenadas saída_facemesh)

        for face_landmarks in saida_facemesh.multi_face_landmarks:
            # desenhando
            # 1 - frame: representa o frame de vídeo
            # 2 - face_landmarks: os landmarks detectados - pontos especificos
            # 3 - FACEMESH_CONTOURS - é uma constante que representa os contornos da malha facial
            mp_drawing.draw_landmarks(frame,face_landmarks,mp_face_mesh.FACEMESH_CONTOURS)


        cv2.imshow('CAmera',frame)
        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

# fecha a captura
cap.release()
cv2.destroyAllWindows()
 
