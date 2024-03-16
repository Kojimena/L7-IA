import numpy as np
import random
import pygame
import sys
import math
from L7viejo import minimax, minimax_pruning
import matplotlib.pyplot as plt



# Configuración del tablero
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

ROW_COUNT = 6
COLUMN_COUNT = 7

VACIO = 0  # Casilla vacia
JUGADOR_PIEZA = 1  # Pieza del jugador
IA_PIEZA = 2  # Pieza de la IA

VENTANA_L = 4  # Longitud de la ventana para ganar

# Parámetros de Q-Learning
ALPHA = 0.1  # Tasa de aprendizaje
GAMMA = 0.9  # Factor de descuento
EPSILON = 0.2  # Probabilidad de exploración

# Inicialización de la tabla Q
Q_table = {}

"""
Modos de Juego
1. IA (TD-Learning) VS IA (Minimax)
2. IA (TD-Learning) VS IA (Minimax con poda alfa-beta)
"""
modo_juego = 1

# Definir jugadores
JUGADOR = 0  # Jugador 1
IA = 1  # Jugador 2

SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
ROW_COUNT = 6
COLUMN_COUNT = 7
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE
size = (width, height)
screen = pygame.display.set_mode(size)

print("Hello TD learning")

def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0


def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def print_board(board):
    print(np.flip(board, 0))


def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][c + 3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][c] == piece:
                return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece:
                return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece:
                return True


def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == 1:
                pygame.draw.circle(screen, RED, (int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == 2:
                pygame.draw.circle(screen, YELLOW, (int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()


"""
Funciones para IA
"""


def evaluar_ventana(ventana, pieza):
    """
    Evalua una ventana, retorna un valor de puntuacion
    """
    score = 0
    oponente = JUGADOR_PIEZA
    if pieza == JUGADOR_PIEZA:
        oponente = IA_PIEZA

    if ventana.count(pieza) == 4:  # 4 en linea
        score += 100
    elif ventana.count(pieza) == 3 and ventana.count(VACIO) == 1:  # 3 en linea
        score += 5
    elif ventana.count(pieza) == 2 and ventana.count(VACIO) == 2:  # 2 en linea
        score += 2

    if ventana.count(oponente) == 3 and ventana.count(VACIO) == 1:  # 3 en linea del oponente
        score -= 4

    return score


def puntuacion_posicion(board, pieza):
    """
    Evalua la puntuacion de un tablero para un jugador dado
    """
    score = 0

    # Puntuacion centro
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(pieza)
    score += center_count * 3

    # Puntuacion Horizontal
    for r in range(ROW_COUNT):
        ventana_row = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            ventana = ventana_row[c:c + VENTANA_L]
            score += evaluar_ventana(ventana, pieza)

    # Puntuacion Vertical
    for c in range(COLUMN_COUNT):
        ventana_col = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            ventana = ventana_col[r:r + VENTANA_L]
            score += evaluar_ventana(ventana, pieza)

    # Puntuacion Diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            ventana = [board[r + i][c + i] for i in range(VENTANA_L)]
            score += evaluar_ventana(ventana, pieza)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            ventana = [board[r + 3 - i][c + i] for i in range(VENTANA_L)]
            score += evaluar_ventana(ventana, pieza)

    return score

def lugares_validos(board):
    """
    Encuentra los lugares validos para dejar una pieza
    """
    lugares_validos = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            lugares_validos.append(col)
    return lugares_validos


def terminal(board):
    """
    Retorna True si el juego ha terminado, False de lo contrario
    """
    return winning_move(board, JUGADOR_PIEZA) or winning_move(board, IA_PIEZA) or len(lugares_validos(board)) == 0


def minimax(tablero, profundidad, maximizando_jugador):
    """
    Calcula el algoritmo minimax para obtener la mejor jugada
    """
    es_terminal =  terminal(tablero)
    if profundidad == 0 or es_terminal:  # profundidad == 0 o juego terminado
        if es_terminal:
            if winning_move(tablero, IA_PIEZA):
                return (None, 100000000000000)
            elif winning_move(tablero, JUGADOR_PIEZA):
                return (None, -10000000000000)
            else: # no hay mas movimientos
                return (None, 0)
        else: # profundidad == 0
            return (None, puntuacion_posicion(tablero, IA_PIEZA))
        
    if maximizando_jugador:
        valor_max = float('-inf')
        columna = random.choice(lugares_validos(tablero))
        for col in lugares_validos(tablero):  # Se prueba cada columna
            fila = get_next_open_row(tablero, col)
            copia_tablero = tablero.copy()
            drop_piece(copia_tablero, fila, col, IA_PIEZA)
            nuevo_valor = minimax(copia_tablero, profundidad-1, False)[1]  # Cambio de jugador
            if nuevo_valor > valor_max:
                valor_max = nuevo_valor
                columna = col
        return columna, valor_max
    else: # Minimizando jugador
        valor_min = float('inf')
        columna = random.choice(lugares_validos(tablero))
        for col in lugares_validos(tablero): # Se prueba cada columna
            fila = get_next_open_row(tablero, col)
            copia_tablero = tablero.copy()
            drop_piece(copia_tablero, fila, col, JUGADOR_PIEZA)
            nuevo_valor = minimax(copia_tablero, profundidad-1, True)[1]  # Cambio de jugador
            if nuevo_valor < valor_min:
                valor_min = nuevo_valor
                columna = col
        return columna, valor_min
        

def minimax_pruning(tablero, profundidad, alfa, beta, maximizando_jugador):
    """
    Calcula el algoritmo minimax con poda alfa-beta para obtener la mejor jugada
    """
    es_terminal =  terminal(tablero)
    if profundidad == 0 or es_terminal:
        if es_terminal:
            if winning_move(tablero, IA_PIEZA):
                return (None, 100000000000000)
            elif winning_move(tablero, JUGADOR_PIEZA):
                return (None, -10000000000000)
            else: # no hay mas movimientos
                return (None, 0)
        else: # profundidad == 0
            return (None, puntuacion_posicion(tablero, IA_PIEZA))
        
    if maximizando_jugador:
        valor_max = float('-inf')
        columna = random.choice(lugares_validos(tablero))
        for col in lugares_validos(tablero):  # Se prueba cada columna
            fila = get_next_open_row(tablero, col)
            copia_tablero = tablero.copy()
            drop_piece(copia_tablero, fila, col, IA_PIEZA)
            nuevo_valor = minimax_pruning(copia_tablero, profundidad-1, alfa, beta, False)[1]
            if nuevo_valor > valor_max:
                valor_max = nuevo_valor
                columna = col
            
            # Poda alfa
            alfa = max(alfa, valor_max)
            if alfa >= beta:  
                break
        return columna, valor_max
    else: # Minimizando jugador
        valor_min = float('inf')
        columna = random.choice(lugares_validos(tablero))
        for col in lugares_validos(tablero):  # Se prueba cada columna
            fila = get_next_open_row(tablero, col)
            copia_tablero = tablero.copy()
            drop_piece(copia_tablero, fila, col, JUGADOR_PIEZA)
            nuevo_valor = minimax_pruning(copia_tablero, profundidad-1, alfa, beta, True)[1]
            if nuevo_valor < valor_min:
                valor_min = nuevo_valor
                columna = col
            
            # Poda beta
            beta = min(beta, valor_min)
            if alfa >= beta:
                break
        return columna, valor_min


# Funciones para Q-Learning
def estado_a_cadena(board):
    return ''.join(str(e) for e in board.reshape(ROW_COUNT * COLUMN_COUNT))


def seleccionar_accion(state, Q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(lugares_validos(state))
    else:
        state_str = estado_a_cadena(state)
        action = np.argmax([Q_table.get((state_str, a), 0) for a in range(COLUMN_COUNT)])
        return action

def actualizar_Q_table(Q_table, state, action, reward, next_state, done):
    state_str = estado_a_cadena(state)
    next_state_str = estado_a_cadena(next_state)
    next_max = np.max([Q_table.get((next_state_str, a), 0) for a in range(COLUMN_COUNT)])
    value = Q_table.get((state_str, action), 0)
    if done:
        Q_table[(state_str, action)] = value + ALPHA * (reward - value)
    else:
        Q_table[(state_str, action)] = value + ALPHA * (reward + GAMMA * next_max - value)

    
def calcular_recompensa(board, piece):
    if winning_move(board, piece):
        return 1  # Ganar
    elif np.all(board != VACIO):  # Tablero lleno
        return 0.5  # Empate
    elif len(lugares_validos(board)) == 0:
        return 0
    else:
        return 0.1

    
def play_step(board, action, piece):
    if not is_valid_location(board, action):
        return board, -1, True
    row = get_next_open_row(board, action)
    drop_piece(board, row, action, piece)
    reward = calcular_recompensa(board, piece)
    return board, reward, terminal(board)

def train_td_learning():
    Q_table = {}
    num_juegos = 1
    for i in range(num_juegos):
        print(f"Juego {i + 1}")
        board = create_board()
        game_over = False
        turno = random.randint(JUGADOR, IA)  # Decide aleatoriamente quién inicia

        while not game_over:
            if turno == JUGADOR:
                # Turno de td learning
                action = seleccionar_accion(board, Q_table, EPSILON)
                next_board, reward, game_over = play_step(board, action, JUGADOR_PIEZA)
                actualizar_Q_table(Q_table, board, action, reward, next_board, game_over)
                board = next_board
            else:
                # Turno de TD Learning
                action = seleccionar_accion(board, Q_table, EPSILON)
                next_board, reward, game_over = play_step(board, action, IA_PIEZA)
                actualizar_Q_table(Q_table, board, action, reward, next_board, game_over)
                board = next_board

            if game_over:
                break

            turno = (turno + 1) % 2  # Cambia el turno

    return Q_table


def jugar_td_vs_minimax(Q_table, num_juegos=10):
    resultados = {'victorias_td': 0, 'victorias_minimax': 0, 'empates': 0}
    for i in range(num_juegos):
        print(f"Juego {i + 1}")
        board = create_board()
        game_over = False
        turno = random.randint(JUGADOR, IA)
        
        if modo_juego == 1:
            while not game_over:
                if turno == JUGADOR:
                    # Turno de TD Learning
                    action = seleccionar_accion(board, Q_table, 0)
                    board, reward, game_over = play_step(board, action, JUGADOR_PIEZA)
                    draw_board(board)
                else:
                    # Turno de Minimax
                    action, _ = minimax(board, 5, True)
                    board, reward, game_over = play_step(board, action, IA_PIEZA)
                    draw_board(board)

                if game_over:
                    if reward == 1:
                        resultados['victorias_td'] += 1
                    elif reward == 0.5:
                        resultados['empates'] += 1
                    else:
                        resultados['victorias_minimax'] += 1
                    break

                turno = (turno + 1) % 2  # Cambia el turno
        elif modo_juego == 2:
            board = create_board()
            game_over = False
            turno = random.randint(JUGADOR, IA)
            while not game_over:
                if turno == JUGADOR:
                    # Turno de TD Learning
                    action = seleccionar_accion(board, Q_table, 0)
                    board, reward, game_over = play_step(board, action, JUGADOR_PIEZA)
                else:
                    # Turno de Minimax con poda alfa-beta
                    action, _ = minimax_pruning(board, 3, -math.inf, math.inf, True)
                    board, reward, game_over = play_step(board, action, IA_PIEZA)

                if game_over:
                    if reward == 1:
                        resultados['victorias_td'] += 1
                    elif reward == 0.5:
                        resultados['empates'] += 1
                    else:
                        resultados['victorias_minimax'] += 1
                    break

                turno = (turno + 1) % 2

    return resultados

def plotear_resultados(resultados):
    fig, ax = plt.subplots()
    ax.bar(resultados.keys(), resultados.values())
    ax.set_xlabel('Jugador')
    ax.set_ylabel('Victorias')
    ax.set_title('Resultados de los juegos')
    plt.show()


if __name__ == "__main__":
    Q_table_entrenada = train_td_learning()  
    resultados_td_vs_minimax = jugar_td_vs_minimax(Q_table_entrenada)
    plotear_resultados(resultados_td_vs_minimax)
    print(resultados_td_vs_minimax)
    

"""
Modos de Juego
1. IA (TD-Learning) VS IA (TD-Learning)
2. IA (TD-Learning) VS IA (Minimax)
"""

""" JUGADOR = 0  # Jugador 1
IA = 1  # Jugador 2
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
ROW_COUNT = 6
COLUMN_COUNT = 7
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE
size = (width, height)
screen = pygame.display.set_mode(size)
pygame.display.update()

#inicializar font
pygame.font.init()

myfont = pygame.font.SysFont("monospace", 75)

# Modo de juego
MODO_JUEGO = 1
game_over = False

while not game_over:
    print("JUGANDO CON TD LEARNING")
    if MODO_JUEGO == 1:
        # IA (TD-Learning) VS IA (TD-Learning)
        if JUGADOR == 0:
            # Entrenar el modelo
            Q_table = train_td_learning()
            print ("Entrenamiento completado")
            # Turno de la IA (TD-Learning)
            action = seleccionar_accion(board, Q_table, 0)
            board, reward, done = play_step(board, action, JUGADOR_PIEZA)
            if done:
                print("Ganó la IA (TD-Learning) 1 ")
                break
        else:
            # Turno de la IA (TD-Learning)
            action = seleccionar_accion(board, Q_table, 0)
            board, reward, done = play_step(board, action, IA_PIEZA)
            if done:
                print("Ganó la IA (TD-Learning) 2")
                break
        print_board(board)
        draw_board(board)
        JUGADOR = (JUGADOR + 1) % 2
    elif MODO_JUEGO == 2:
        # IA (TD-Learning) VS IA (Minimax)
        if JUGADOR == 0:
            # Turno de la IA (TD-Learning)
            action = seleccionar_accion(board, Q_table, 0)
            board, reward, done = play_step(board, action, JUGADOR_PIEZA)
            if done:
                print("Ganó la IA (TD-Learning)")
                break
        else:
            # Turno de la IA (Minimax)
            action, _ = minimax(board, 3, -math.inf, math.inf, True)
            board, reward, done = play_step(board, action, IA_PIEZA)
            if done:
                print("Ganó la IA (Minimax)")
                break
        print_board(board)
        draw_board(board)
        JUGADOR = (JUGADOR + 1) % 2
    elif MODO_JUEGO == 3:
        # IA (TD-Learning) VS IA (Minimax con poda alfa-beta)
        if JUGADOR == 0:
            # Turno de la IA (TD-Learning)
            action = seleccionar_accion(board, Q_table, 0)
            board, reward, done = play_step(board, action, JUGADOR_PIEZA)
            if done:
                print("Ganó la IA (TD-Learning)")
                break
        else:
            # Turno de la IA (Minimax con poda alfa-beta)
            action, _ = minimax_pruning(board, 3, -math.inf, math.inf, True)
            board, reward, done = play_step(board, action, IA_PIEZA)
            if done:
                print("Ganó la IA (Minimax con poda alfa-beta)")
                break
        print_board(board)
        draw_board(board)
        JUGADOR = (JUGADOR + 1) % 2
 """