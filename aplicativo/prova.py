import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import os
import sys

# Inicializa o mixer de áudio
pygame.mixer.init()

# Carrega os arquivos de som
pygame.mixer.music.load("alarmenavio.mp3")  # Alarme da boca
som_olho = pygame.mixer.Sound("businadj.mp3")  # Alarme dos olhos
som_assobio = pygame.mixer.Sound("assobio.mp3")  # Som para rosto detectado

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
mar_limiar = 0.5
dormindo = False  # Flag para controle dos olhos fechados
aberto_boca = False  # Flag para controle da boca aberta
tempo_olhos_fechados = 0.0  # Tempo que os olhos ficaram fechados
tempo_boca_aberta = 0.0  # Tempo que a boca ficou aberta
contagem_piscadas = 0  # Contagem de piscadas

# Inicializa a câmera
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Estado do som
som_boca_tocando = False
som_olho_tocando = False
som_assobio_tocando = False

# Variáveis para controle de exibição das mensagens
mensagem_boca = ""
mensagem_olhos = ""
tempo_inicio_mensagem_boca = None
tempo_inicio_mensagem_olhos = None
duracao_mensagem = 3  # Duração das mensagens em segundos

# Define o tamanho da janela
window_width = 800
window_height = 600

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera', window_width, window_height)

# Variáveis de tempo para o aviso de rosto
rosto_detectado = False
tempo_inicial_aviso = None

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
            rosto_detectado = True
            tempo_inicial_aviso = time.time()   

            for face_landmarks in saida_facemesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 220), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 204, 0), thickness=1, circle_radius=1)
                )
                
                face = face_landmarks.landmark
                
                ear = calculo_ear(face, p_olho_dir, p_olho_esq)
                mar = calculo_mar(face, p_boca)
                cv2.rectangle(frame,(0,1),(290,140),(58,58,55),-1)

                # Verificação da condição da boca aberta
                if mar > mar_limiar:
                    if not aberto_boca:
                        tempo_inicial_boca_aberta = time.time()
                        aberto_boca = True
                    else:
                        tempo_boca_aberta = time.time() - tempo_inicial_boca_aberta
                    estado_boca = "aberta"
                else:
                    tempo_boca_aberta = 0.0
                    aberto_boca = False
                    estado_boca = "fechada"

                # Verificação da condição dos olhos fechados
                if ear < ear_limiar:
                    if not dormindo:
                        tempo_inicial_olhos_fechados = time.time()
                        dormindo = True
                    else:
                        tempo_olhos_fechados = time.time() - tempo_inicial_olhos_fechados
                    estado_olho = "fechados"
                else:
                    if dormindo:
                        contagem_piscadas += 1
                    tempo_olhos_fechados = 0.0
                    dormindo = False
                    estado_olho = "abertos"

                # Exibir estados e tempos na tela
                cv2.rectangle(frame, (0, 0), (290, 100), (58, 58, 55), -1)  # Fundo do quadro
                cv2.putText(frame, f"MAR - Boca: {estado_boca} - {round(tempo_boca_aberta, 2)}s", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"EAR - Olhos: {estado_olho} - {round(tempo_olhos_fechados, 2)}s", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Piscadas: {contagem_piscadas}", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


                # Alarme para boca aberta
                if tempo_boca_aberta >= 1.5 and not som_boca_tocando:
                    mensagem_boca = f"Alerta de sono, bocejando!"
                    tempo_inicio_mensagem_boca = time.time()
                    pygame.mixer.music.play(-1)
                    som_boca_tocando = True
                elif tempo_boca_aberta == 0.0 and som_boca_tocando:
                    pygame.mixer.music.stop()
                    som_boca_tocando = False

                # Alarme para olhos fechados
                if tempo_olhos_fechados >= 1.0 and not som_olho_tocando:
                    mensagem_olhos = "Olhos fechados por muito tempo!"
                    tempo_inicio_mensagem_olhos = time.time()
                    som_olho.play()
                    som_olho_tocando = True
                elif tempo_olhos_fechados == 0.0 and som_olho_tocando:
                    som_olho.stop()
                    som_olho_tocando = False

                # Exibir mensagens no canto inferior da tela
                altura, largura, _ = frame.shape
                if mensagem_boca and (time.time() - tempo_inicio_mensagem_boca <= duracao_mensagem):
                    cv2.rectangle(frame, (0, altura - 70), (largura, altura), (0, 0, 0), -1)  # Fundo preto
                    cv2.putText(frame, mensagem_boca, (10, altura - 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
                else:
                    mensagem_boca = ""  # Limpa a mensagem após a duração

                if mensagem_olhos and (time.time() - tempo_inicio_mensagem_olhos <= duracao_mensagem):
                    cv2.rectangle(frame, (0, altura - 140), (largura, altura - 70), (0, 0, 0), -1)  # Fundo preto
                    cv2.putText(frame, mensagem_olhos, (10, altura - 100), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
                else:
                    mensagem_olhos = ""  # Limpa a mensagem após a duração

        else:
            # Atualiza o estado e tempo quando nenhum rosto é detectado
            if rosto_detectado:
                tempo_inicial_aviso = time.time()
            rosto_detectado = False

        # Exibe aviso de rosto detectado ou não detectado no canto superior direito por 5 segundos
        if tempo_inicial_aviso and (time.time() - tempo_inicial_aviso <= 5):
            mensagem = "Rosto detectado" if rosto_detectado else "Nenhum rosto detectado"
            cor_texto = (0, 255, 0) if rosto_detectado else (0, 0, 255)  # Verde para detectado, vermelho para não detectado
            largura_frame = frame.shape[1]
            tamanho_texto = cv2.getTextSize(mensagem, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]  # Obtém o tamanho do texto
            posicao_x = largura_frame - tamanho_texto[0] - 10  # Calcula a posição x do texto para o canto direito
            posicao_y = 30  # Altura fixa no canto superior
            cv2.putText(frame, mensagem, (posicao_x, posicao_y), cv2.FONT_HERSHEY_DUPLEX, 1.0, cor_texto, 2)


         # Exibe aviso de rosto detectado ou não detectado no canto superior direito por 5 segundos
        if tempo_inicial_aviso and (time.time() - tempo_inicial_aviso <= 5):
            mensagem = "Rosto detectado" if rosto_detectado else "Nenhum rosto detectado"
            cor_texto = (0, 255, 0) if rosto_detectado else (0, 0, 255)  # Verde para detectado, vermelho para não detectado
            largura_frame = frame.shape[1]
            tamanho_texto = cv2.getTextSize(mensagem, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]  # Obtém o tamanho do texto
            posicao_x = largura_frame - tamanho_texto[0] - 10  # Calcula a posição x do texto para o canto direito
            posicao_y = 30  # Altura fixa no canto superior
            cv2.putText(frame, mensagem, (posicao_x, posicao_y), cv2.FONT_HERSHEY_DUPLEX, 1.0, cor_texto, 2)

            # Tocar som de assobio quando rosto detectado
            if rosto_detectado and not som_assobio_tocando:
                som_assobio.play()
                som_assobio_tocando = True
        else:
            # Para o som quando o tempo de exibição da mensagem acaba
            if som_assobio_tocando:
                som_assobio.stop()
                som_assobio_tocando = False

        cv2.imshow('Camera', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()