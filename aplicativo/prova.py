import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import os
import sys

# Inicializa o mixer de áudio
pygame.mixer.init()
# Carrega o arquivo de som (Alarme)
pygame.mixer.music.load("alarmenavio.mp3")

# Pontos dos olhos e boca
p_olho_esq = [385, 380, 387, 373, 362, 263]
p_olho_dir = [160, 144, 158, 153, 33, 133]
p_olhos = p_olho_esq + p_olho_dir

p_boca = [82, 87, 13, 14, 312, 317, 78, 308]

# Função EAR (Eye Aspect Ratio)
def calculo_ear(face, p_olho_dir, p_olho_esq):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_esq = face[p_olho_esq, :]
        face_dir = face[p_olho_dir, :]

        ear_esq = (np.linalg.norm(face_esq[0] - face_esq[1]) + np.linalg.norm(face_esq[2] - face_esq[3])) / (2 * (np.linalg.norm(face_esq[4] - face_esq[5])))

        ear_dir = (np.linalg.norm(face_dir[0] - face_dir[1]) + np.linalg.norm(face_dir[2] - face_dir[3])) / (2 * (np.linalg.norm(face_dir[4] - face_dir[5])))

    except:
        ear_esq = 0.0
        ear_dir = 0.0
    media_ear = (ear_esq + ear_dir) / 2
    return media_ear

# Função MAR (Mouth Aspect Ratio)
def calculo_mar(face, p_boca):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_boca = face[p_boca, :]

        mar = (np.linalg.norm(face_boca[0] - face_boca[1]) + np.linalg.norm(face_boca[2] - face_boca[3]) + np.linalg.norm(face_boca[4] - face_boca[5])) / (2 * (np.linalg.norm(face_boca[6] - face_boca[7])))
    except:
        mar = 0.0
    return mar

# Limiares
ear_limiar = 0.27
mar_limiar = 0.1
dormindo = 0  # Flag para controle dos olhos fechados
aberto_boca = False  # Flag para controle da boca aberta
tempo_olhos_fechados = 0.0  # Tempo que os olhos ficaram fechados
tempo_boca_aberta = 0.0  # Tempo que a boca ficou aberta

# Inicializa a câmera
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Estado do som
som_tocando = False

# Define o tamanho da janela
window_width = 800
window_height = 600

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera', window_width, window_height)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            print('Ignorando o frame vazio da câmera.')
            continue
        
        comprimento, largura, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        saida_facemesh = facemesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if saida_facemesh.multi_face_landmarks:
            # Detectando rosto
            print("Rosto detectado")
            if not som_tocando:
                pygame.mixer.music.play(-1)  # Toca continuamente
                som_tocando = True  # Atualiza o estado para som tocando
        else:
            print("Nenhum rosto detectado")
            if som_tocando:
                pygame.mixer.music.stop()  # Para o som
                som_tocando = False  # Atualiza o estado para som parado

        try:
            for face_landmarks in saida_facemesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 102, 102), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(102, 204, 0), thickness=1, circle_radius=1)
                )
                
                face = face_landmarks.landmark
                
                for id_coord, coord_xyz in enumerate(face):
                    if id_coord in p_olhos:
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, largura, comprimento)
                        cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)
                    if id_coord in p_boca:
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, largura, comprimento)
                        cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)

                # Chamada do EAR e print
                ear = calculo_ear(face, p_olho_dir, p_olho_esq)
                cv2.rectangle(frame, (0, 1), (290, 140), (58, 58, 55), -1)
                cv2.putText(frame, f"EAR: {round(ear, 2)}", (1, 24),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.9, (255, 255, 255), 2)

                # Chamada do MAR e print
                mar = calculo_mar(face, p_boca)
                cv2.putText(frame, f"MAR: {round(mar, 2)} { 'abertos' if mar >= mar_limiar else  'fechados '}", (1, 50),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.9, (255, 255, 255), 2)

                # Verificação da condição dos olhos fechados
                if ear < ear_limiar:
                    if dormindo == 0:
                        t_inicial = time.time()  # Marca o tempo inicial
                    dormindo = 1  # Olhos estão fechados
                if dormindo == 1 and ear >= ear_limiar:
                    dormindo = 0  # Olhos abriram
                t_final = time.time()

                tempo = (t_final - t_inicial) if dormindo == 1 else 0.0
                tempo_olhos_fechados = tempo
                cv2.putText(frame, f"Tempo olhos fechados: {round(tempo_olhos_fechados, 3)}", (1, 80),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.9, (255, 255, 255), 2)
                
                # Exibe alerta se os olhos ficarem fechados por mais de 1.5 segundos
                if tempo_olhos_fechados >= 1.5:
                    cv2.rectangle(frame, (30, 400), (610, 452), (109, 233, 219), -1)
                    cv2.putText(frame, f"Alerta: Olhos fechados por muito tempo!", (80, 435),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.85, (58, 58, 55), 1)
                    if not som_tocando:
                        pygame.mixer.music.play(-1)  # Toca o som de alerta
                        som_tocando = True

                # Verificação da condição da boca aberta
                if mar > mar_limiar:
                    if not aberto_boca:
                        tempo_boca_aberta = time.time()  # Marca o tempo inicial da boca aberta
                        aberto_boca = True
                else:
                    if aberto_boca:
                        tempo_boca_aberta = 0.0  # Resetar tempo da boca fechada
                        aberto_boca = False

                # Exibe o tempo da boca aberta
                cv2.putText(frame, f"Tempo boca aberta: {round(tempo_boca_aberta, 3)}", (1, 110),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.9, (55, 255, 255), 2)
                
        except Exception as e:
            print("Erro:", e)

        # Exibindo o frame com as marcações
        cv2.imshow('Camera', frame)

        # Se pressionar 'c', fecha a janela
        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

# Finaliza a captura da câmera e fecha a janela
cap.release()
cv2.destroyAllWindows()

