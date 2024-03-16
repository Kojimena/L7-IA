import numpy as np
import pygame
import sys
import math
import random
import matplotlib.pyplot as plt

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

ROW_COUNT = 6
COLUMN_COUNT = 7

JUGADOR = 0  # Jugador 1
IA = 1  # Jugador 2

VACIO = 0  # Casilla vacia
JUGADOR_PIEZA = 1  # Pieza del jugador
IA_PIEZA = 2  # Pieza de la IA

VENTANA_L = 4  # Longitud de la ventana para ganar


def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board


def drop_piece(board, row, col, piece):
    board[row][col] = piece
    return board


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
    es_terminal = terminal(tablero)
    if profundidad == 0 or es_terminal:  # profundidad == 0 o juego terminado
        if es_terminal:
            if winning_move(tablero, IA_PIEZA):
                return (None, 100000000000000)
            elif winning_move(tablero, JUGADOR_PIEZA):
                return (None, -10000000000000)
            else:  # no hay mas movimientos
                return (None, 0)
        else:  # profundidad == 0
            return (None, puntuacion_posicion(tablero, IA_PIEZA))

    if maximizando_jugador:
        valor_max = float('-inf')
        columna = random.choice(lugares_validos(tablero))
        for col in lugares_validos(tablero):  # Se prueba cada columna
            fila = get_next_open_row(tablero, col)
            copia_tablero = tablero.copy()
            drop_piece(copia_tablero, fila, col, IA_PIEZA)
            nuevo_valor = minimax(copia_tablero, profundidad - 1, False)[1]  # Cambio de jugador
            if nuevo_valor > valor_max:
                valor_max = nuevo_valor
                columna = col
        return columna, valor_max
    else:  # Minimizando jugador
        valor_min = float('inf')
        columna = random.choice(lugares_validos(tablero))
        for col in lugares_validos(tablero):  # Se prueba cada columna
            fila = get_next_open_row(tablero, col)
            copia_tablero = tablero.copy()
            drop_piece(copia_tablero, fila, col, JUGADOR_PIEZA)
            nuevo_valor = minimax(copia_tablero, profundidad - 1, True)[1]  # Cambio de jugador
            if nuevo_valor < valor_min:
                valor_min = nuevo_valor
                columna = col
        return columna, valor_min


def minimax_pruning(tablero, profundidad, alfa, beta, maximizando_jugador):
    """
    Calcula el algoritmo minimax con poda alfa-beta para obtener la mejor jugada
    """
    es_terminal = terminal(tablero)
    if profundidad == 0 or es_terminal:
        if es_terminal:
            if winning_move(tablero, IA_PIEZA):
                return (None, 100000000000000)
            elif winning_move(tablero, JUGADOR_PIEZA):
                return (None, -10000000000000)
            else:  # no hay mas movimientos
                return (None, 0)
        else:  # profundidad == 0
            return (None, puntuacion_posicion(tablero, IA_PIEZA))

    if maximizando_jugador:
        valor_max = float('-inf')
        columna = random.choice(lugares_validos(tablero))
        for col in lugares_validos(tablero):  # Se prueba cada columna
            fila = get_next_open_row(tablero, col)
            copia_tablero = tablero.copy()
            drop_piece(copia_tablero, fila, col, IA_PIEZA)
            nuevo_valor = minimax_pruning(copia_tablero, profundidad - 1, alfa, beta, False)[1]
            if nuevo_valor > valor_max:
                valor_max = nuevo_valor
                columna = col

            # Poda alfa
            alfa = max(alfa, valor_max)
            if alfa >= beta:
                break
        return columna, valor_max
    else:  # Minimizando jugador
        valor_min = float('inf')
        columna = random.choice(lugares_validos(tablero))
        for col in lugares_validos(tablero):  # Se prueba cada columna
            fila = get_next_open_row(tablero, col)
            copia_tablero = tablero.copy()
            drop_piece(copia_tablero, fila, col, JUGADOR_PIEZA)
            nuevo_valor = minimax_pruning(copia_tablero, profundidad - 1, alfa, beta, True)[1]
            if nuevo_valor < valor_min:
                valor_min = nuevo_valor
                columna = col

            # Poda beta
            beta = min(beta, valor_min)
            if alfa >= beta:
                break
        return columna, valor_min


"""
TD Learning
"""

def get_state(board):
    """
    Representación del estado
    :param board: Tablero de juego
    :return: Representación del estado
    """
    # board, turno, ganador

    # Codificar el tablero de juego
    game_state = []
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            if board[r][c] == JUGADOR_PIEZA:
                game_state.append(1)  # Casilla ocupada por el jugador 1
            elif board[r][c] == IA_PIEZA:
                game_state.append(2)  # Casilla ocupada por el jugador 2
            else:
                game_state.append(0)  # Casilla vacía

    # Agregar información sobre el turno del jugador
    if turn == JUGADOR:
        game_state.append(1)  # Turno del jugador 1
    else:
        game_state.append(2)  # Turno del jugador 2

    # Agregar información sobre el estado del juego (si ha terminado y quién ha ganado)
    if winning_move(board, JUGADOR_PIEZA):
        game_state.append(1)  # El jugador 1 ha ganado
    elif winning_move(board, IA_PIEZA):
        game_state.append(-1)  # El jugador 2 ha ganado
    elif len(lugares_validos(board)) == 0:
        game_state.append(0)  # Empate
    else:
        game_state.append(0)  # Juego en curso

    return game_state

def get_action_space(state):
    """
    Espacio de acción
    :param state: Representación del estado
    :return: Espacio de acción
    """

    dim = ROW_COUNT * COLUMN_COUNT
    board_from_state = np.array(state[:dim]).reshape(ROW_COUNT, COLUMN_COUNT)
    return lugares_validos(board_from_state)


def decode_board_state(game_state):
    board = []
    for i in range(ROW_COUNT):
        row = []
        for j in range(COLUMN_COUNT):
            index = i * COLUMN_COUNT + j
            if game_state[index] == 1:
                row.append(JUGADOR_PIEZA)
            elif game_state[index] == -1:
                row.append(IA_PIEZA)
            else:
                row.append(VACIO)
        board.append(row)
    return board


def get_reward(state, action, player): 
    """"
    Recompensas
    :param state: Representación del estado
    :param action: Acción
    :param player: Jugador
    """
    board_from_state = decode_board_state(state)

    next_board = board_from_state.copy()
    row = get_next_open_row(next_board, action)
    drop_piece(next_board, row, action, JUGADOR_PIEZA)

    other_player = (player + 1) % 2

    if winning_move(next_board, JUGADOR_PIEZA if player == JUGADOR else IA_PIEZA):
        return 100
    elif winning_move(next_board, JUGADOR_PIEZA if other_player == JUGADOR else IA_PIEZA):
        return -100
    elif len(lugares_validos(next_board)) == 0:
        return -10
    else:
        return 1


def get_next_state(state, action, player):  # Se define el siguiente estado como el estado resultante de dejar una ficha en una columna
    dim = ROW_COUNT * COLUMN_COUNT
    board_from_state = decode_board_state(state)
    next_board = board_from_state.copy()

    row = get_next_open_row(next_board, action)
    drop_piece(next_board, row, action, JUGADOR_PIEZA if player == JUGADOR else IA_PIEZA)

    next_state = get_state(next_board)

    return next_state


def get_initial_state():  # Se define el estado inicial como el tablero vacío
    return get_state(create_board())


def is_terminal_state(state):  # Se define un estado terminal como un estado en el que el juego ha terminado
    return state[-1] != 0


def get_winner(state):  # Se define el ganador como el jugador que ha ganado
    if state[-1] == 1:
        return 1
    elif state[-1] == -1:
        return -1
    else:
        return 0


def get_state_representation(state):  # Se define la representación del estado como el tablero de juego
    dim = ROW_COUNT * COLUMN_COUNT
    return np.array(state[:dim]).reshape(ROW_COUNT, COLUMN_COUNT)


def td_learning(Q, state, action, reward, next_state, alpha, gamma):
    """
    Algoritmo de aprendizaje TD
    :param Q: Modelo
    :param state: Estado
    :param action: Acción
    :param reward: Recompensa
    :param next_state: Siguiente estado
    :param alpha: Tasa de aprendizaje
    :param gamma: Factor de descuento
    :return: Modelo actualizado
    """

    dim = ROW_COUNT * COLUMN_COUNT
    state_representation = np.array(state[:dim]).reshape(ROW_COUNT, COLUMN_COUNT)
    next_state_representation = np.array(next_state[:dim]).reshape(ROW_COUNT, COLUMN_COUNT)
    state_action = (tuple(state[:dim]), action)
    next_action_space = get_action_space(next_state)
    next_action = random.choice(next_action_space)
    next_state_action = (tuple(next_state[:dim]), next_action)

    if next_state_action not in Q:
        Q[next_state_action] = 0

    if state_action not in Q:
        Q[state_action] = 0

    # Actualización de valor
    Q[state_action] = Q[state_action] + alpha * (reward + gamma * Q[next_state_action] - Q[state_action])

    return Q


def train_td_learning(episodes):  # Se entrena el modelo usando el algoritmo TD-learning
    Q = {}
    alpha = 0.1
    gamma = 0.8
    epsilon = 0.9 # estrategia de exploración epsilon-greedy 
    decay = 0.9999

    print("Entrenando el modelo TD...")
    for episode in range(episodes):
        state = get_initial_state()
        turn = random.choice([JUGADOR, IA])

        while not is_terminal_state(state):
            if turn == JUGADOR:  # Jugador a entrenar
                action_space = get_action_space(state)
                
                # epsilon-greedy
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(action_space)
                else:
                    action = max(action_space, key=lambda x: Q.get((tuple(state[:ROW_COUNT * COLUMN_COUNT]), x), 0))

                reward = get_reward(state, action, turn)

                next_state = get_next_state(state, action, turn)
                Q = td_learning(Q, state, action, reward, next_state, alpha, gamma)
                state = next_state
                turn += 1
                turn = turn % 2

            else:
                # este es cualquier jugador que no sea el jugador a entrenar. elige una acción aleatoria
                action_space = get_action_space(state)
                action = random.choice(action_space)
                next_state = get_next_state(state, action, turn)
                state = next_state

                turn += 1
                turn = turn % 2

        epsilon *= decay
        if episode % 1000 == 0:
            print("Episodio", episode)

    print("Modelo TD entrenado")
    return Q


def play_td_vs_minimax(Q, algoritmo_alfa_beta=False, profundidad=5):
    """
    Función para enfrentar el modelo TD contra el algoritmo minimax
    :param Q: Modelo TD
    :param algoritmo_alfa_beta: Si es True, se usa el algoritmo minimax con poda alfa-beta
    """

    board = create_board()
    print_board(board)
    game_over = False
    turn = random.choice([JUGADOR, IA])  # Se elige aleatoriamente quién empieza

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        if turn == JUGADOR and not game_over:  # Algoritmo de TD
            state = get_state(board)
            action_space = get_action_space(state)

            action = max(action_space, key=lambda x: Q.get((tuple(state[:ROW_COUNT * COLUMN_COUNT]), x), 0))

            if is_valid_location(board, action):
                row = get_next_open_row(board, action)
                drop_piece(board, row, action, JUGADOR_PIEZA)

                if winning_move(board, JUGADOR_PIEZA):
                    print("Algoritmo de TD gana!!")
                    game_over = True
                    
                    draw_board(board)
                    print_board(board)

                    return JUGADOR

                turn += 1
                turn = turn % 2

        if turn == IA and not game_over:  # Algoritmo de minimax
            if algoritmo_alfa_beta:
                action, _ = minimax_pruning(board, profundidad, float('-inf'), float('inf'), False)
            else:
                action, _ = minimax(board, profundidad, False)
            if is_valid_location(board, action):
                row = get_next_open_row(board, action)
                drop_piece(board, row, action, IA_PIEZA)

                if winning_move(board, IA_PIEZA):
                    print("Algoritmo de minimax gana!!")
                    game_over = True

                    print_board(board)
                    draw_board(board)

                    return IA

                turn += 1
                turn = turn % 2


board = create_board()
print_board(board)
game_over = False
turn = 0

Q_trained = train_td_learning(100_000) #Ciclo de entrenamiento
print(Q_trained)

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE / 2 - 5)

screen = pygame.display.set_mode(size)
draw_board(board)
pygame.display.update()

myfont = pygame.font.SysFont("monospace", 75)

"""
Modos de Juego TD vs Minimax

Se juegan 75 partidas entre el modelo TD y el algoritmo minimax. Se grafican los resultados.
El algoitmo minimax puede ser con o sin poda alfa-beta. La profundidad del árbol de búsqueda puede ser modificada.

"""

IA2_Pruning = False
IA2_Profundidad = 5
JUEGOS = 75

ganadores = []
for i in range(JUEGOS):
    print("Jugando TD vs Minimax ", i)
    ganadores.append(play_td_vs_minimax(Q_trained, IA2_Pruning, IA2_Profundidad))
    game_over = True

# imprimir los resultados como tabla de frecuencias
print("Frecuencia de victorias")
print("TD: ", ganadores.count(JUGADOR))
print("Minimax: ", ganadores.count(IA))

# graficar los resultados
etiquetas = ['TD', 'Minimax']
frecuencias = [ganadores.count(JUGADOR), ganadores.count(IA)]
plt.bar(etiquetas, frecuencias)
plt.show()