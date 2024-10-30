import cv2

# capturar a câmera
cap = cv2.VideoCapture(0)
# while cap.isOpened() - enquanto a camera estiver aberta o while continuara rodando
while cap.isOpened():
    # Sucesso é booleana, significa que se identificar algum vídeo ou frame, retornará verdadeiro. Se não identificar nenhum vídeo, retornará falso
    # FRAME é uma variável frame que por sua vez, é a captura coletada pela câmera. 
    sucesso,frame = cap.read() # Utilizaremos o método cap.read() para leitura
    if not sucesso:
        print('Ignorando o frame vazio da câmera')
        continue # pesquise sobre a diferença do continue e break (história go to)
    # depois disso, saímos do bloco condicional e podemos visualizar a captura e conferir o que está acontecendo.
    # podemos contar com o método imshow( ) pra isso
    # dentro dos parenteses passamos dois parâmetros 'Camera' e frame
    # Passaremos um nome para nossa janela no formato string 'Camera'
    # Também outro parametro - frame - definindo que o que queremos mostrar que é nosso próprio frame
    cv2.imshow('Camera',frame)
    # com isso, conseguiremos mostrar a imagem, mas não temos uma opção de controle para fechar a camera em tempo real.
    # Para isso, usaremos um condicional if
    # Ele vai pegar o laço de repetição quando apertarmos teclas específicas.
    # Como a intenção é fechar vamos usar "C" de "Close"
    # waitKey(10) vai esperar uma chave
    # verificar se essa chave é c
    # Para isso passamos & OxFF
    """
      Detalhando mais

      cv2.waitKey(10)

            cv2.waitKey() é uma função do OpenCV
            Que espera por uma entrada de tecla por um certo período de tempo em milissegundos
            Passado como argumento. 
            Nesse caso, 10 indica que o programa espera 10 milissegundos.
            Se uma tecla for pressionada durante esse tempo o código da tecla será retornado.
            Se nenhum valor for passado, o waitKey() não bloqueia a execução e retorna -1 se nenhuma tecla for pressionada
    
      & 0xFF

            O operador & 0xFF é uma operação de bitwise AND com 0xFF (255 em hexadecimal).
            Isso é feito para garantir compatibilidade com sistemas operacionais diferentes.
            Isolando o último byte do código da tecla pressionada. 
            Muitas vezes, sistemas operacionais retornam um valor de 32 bits.
            Mas 0xFF restringe o valor à faixa de 0 a 255, que é a faixa da tabela ASCII.    
    
      == ord('c')

           ord('c') retorna o código ASCII do caractere 'c', que é 99.
           Esse trecho está verificando se a tecla pressionada foi 'c'. 
           Se for, a condição do if será verdadeira, permitindo que um bloco de código seja 
           executado a seguir.
    """
    if cv2.waitKey(10) & 0xFF == ord('c'):
        break
# agora para fechar a caputura fora do if e do while
cap.release()
# também, vamos fechar todas as janelas/pop-ups, pois o método imshow() vai abrir um pop-up, mostrando um frame
cv2.destroyAllWindows()
# agora é só conectar a camera e executar