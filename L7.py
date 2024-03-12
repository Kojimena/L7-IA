import numpy as np
import pygame
import sys
import math
import random

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
    

board = create_board()
print_board(board)
game_over = False
turn = 0

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
Modos de Juego
1. IA vs Jugador
2. IA vs IA
"""
MODO_JUEGO = 2

"""
Algoritmo de IA
1. Minimax
2. Minimax con poda alfa-beta
"""

ALGORITMO_IA = 1
ALGORITMO_IA2 = 2

while not game_over:

    if MODO_JUEGO == 1:   # IA vs Jugador
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                if turn == JUGADOR:
                    pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)

            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                # print(event.pos)
                # Jugador 1 input
                if turn == JUGADOR:
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))

                    if is_valid_location(board, col):
                        row = get_next_open_row(board, col)
                        drop_piece(board, row, col, JUGADOR_PIEZA)

                        if winning_move(board, JUGADOR_PIEZA):
                            label = myfont.render("Jugador 1 gana!!", 1, RED)
                            screen.blit(label, (40, 10))
                            game_over = True

                        turn += 1
                        turn = turn % 2

                        print_board(board)
                        draw_board(board)

            if turn == IA and not game_over:
                #col = random.randint(0, COLUMN_COUNT-1)
                if ALGORITMO_IA == 1:
                    col, minimax_score = minimax(board, 5, True)
                else:
                    col, minimax_score = minimax_pruning(board, 5, float('-inf'), float('inf'), True)

                if is_valid_location(board, col):
                    pygame.time.wait(500)
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, IA_PIEZA)

                    if winning_move(board, IA_PIEZA):
                        label = myfont.render("IA gana!!", 1, YELLOW)
                        screen.blit(label, (40, 10))
                        game_over = True

                    print_board(board)
                    draw_board(board)

                    turn += 1
                    turn = turn % 2

                    if game_over:
                        pygame.time.wait(3000)

    elif MODO_JUEGO == 2:   # IA vs IA
        # La segunda IA controlará la pieza Jugador
        if turn == JUGADOR and not game_over:  # IA 1
            if ALGORITMO_IA == 1:
                col, minimax_score = minimax(board, 1, True)
            else:
                col, minimax_score = minimax_pruning(board, 1, float('-inf'), float('inf'), True)

            if is_valid_location(board, col):
                pygame.time.wait(500)
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, JUGADOR_PIEZA)

                if winning_move(board, JUGADOR_PIEZA):
                    label = myfont.render("IA 1 gana!!", 1, RED)
                    screen.blit(label, (40, 10))
                    game_over = True

                print_board(board)
                draw_board(board)

                turn += 1
                turn = turn % 2

                if game_over:
                    pygame.time.wait(3000)

        if turn == IA and not game_over:  # IA 2
            if ALGORITMO_IA2 == 1:
                col, minimax_score = minimax(board, 5, False)
            else:
                col, minimax_score = minimax_pruning(board, 5, float('-inf'), float('inf'), False)

            if is_valid_location(board, col):
                pygame.time.wait(500)
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, IA_PIEZA)

                if winning_move(board, IA_PIEZA):
                    label = myfont.render("IA 2 gana!!", 1, YELLOW)
                    screen.blit(label, (40, 10))
                    game_over = True

                print_board(board)
                draw_board(board)

                turn += 1
                turn = turn % 2

                if game_over:
                    pygame.time.wait(3000)
