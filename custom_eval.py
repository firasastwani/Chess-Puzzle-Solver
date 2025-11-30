import chess

"""
Custom evaluation functions - pure implementations without external engines.
All evaluation is done using custom heuristics and chess logic.
"""

# ==================== EVALUATION FUNCTIONS ====================

def evaluate_position(board):
    """"
    Returns a score from the perspective of the current player. 

    Positive = good for white, Negative = good for black

    Returns: 
        int: Position score (in centipawns) 100 = 1 pawn advantage
    
    """""
# worst case for evaluated side (white)
    if board.is_checkmate():
        return -100000
    
    # neutral, game ends in a draw
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    # give piece values to each piece
    piece_values = {
        chess.PAWN: 100, 
        chess.KNIGHT: 300, 
        chess.BISHOP: 300,
        chess.ROOK: 500, 
        chess.QUEEN: 900, 
        chess.KING: 0 # handeled elsewhere
    }
    
    score = 0

    # Material count WITH hanging piece detection
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        piece_value = piece_values.get(piece.piece_type, 0)
        
        # Check if piece is hanging (attacked and not defended)
        is_attacked_by_opponent = board.is_attacked_by(not piece.color, square)
        is_defended = board.is_attacked_by(piece.color, square)
        
        if is_attacked_by_opponent and not is_defended:
            # Piece is hanging! Apply huge penalty/bonus
            if piece.color == chess.WHITE:
                score -= piece_value * 2  # White's hanging piece - bad for white
            else:
                score += piece_value * 2  # Black's hanging piece - good for white
        
        # Regular material count
        if piece.color == chess.WHITE:
            score += piece_value
        else:
            score -= piece_value
    
    # Add positional bonuses (reduce impact, less important than material)
    score += evaluate_piece_positions(board) // 10
    
    # Add king safety evaluation
    score += evaluate_king_saftey(board)
    
    #Bonus for checking the opponent
    if board.is_check():
        score += 500  # Checks are very valuable in mate puzzles
    
    # Bonus for attacking squares near opponent's king
    score += evaluate_king_attacks(board) // 5  # Reduce impact
    
    # Add mobility (number of legal moves is good)
    mobility = len(list(board.legal_moves))
    score += mobility * 5
    
    # Return from white's perspective
    return score if board.turn == chess.WHITE else -score


def evaluate_position_fast(board):
    """
    Fast material evaluation with tactical bonuses for minimax search.
    Stockfish is too slow to call thousands of times!
    
    Args:
        board: chess.Board position
    
    Returns:
        int: Position evaluation in centipawns (positive = white advantage)
    """
    if board.is_checkmate():
        return -100000
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    # Simple material count (fast!)
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }
    
    score = 0
    for piece_type in piece_values:
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        score += (white_count - black_count) * piece_values[piece_type]
    
    # CRITICAL: Bonus for checks (very important for mate puzzles!)
    if board.is_check():
        score += 1000 if board.turn == chess.WHITE else -1000  # Increased from 500
    
    # Bonus for moves that give check (helps find mating sequences)
    # This is evaluated when we look at moves, not positions
    
    # Penalty for being in check (bad for the side to move)
    if board.is_check():
        score -= 200 if board.turn == chess.WHITE else 200
    
    return score if board.turn == chess.WHITE else -score


def evaluate_king_attacks(board):
    """
    Evaluate attacks near the opponent's king
    """
    score = 0

    opponent_color = not board.turn
    king_square = board.king(opponent_color)

    if king_square is None: 
        return 0
    
    king_zone = []
    king_rank = chess.square_rank(king_square)
    king_file = chess.square_file(king_square)

    for rank_offset in [-1, 0, 1]:
        for file_offset in [-1, 0, 1]:
            new_rank = king_rank + rank_offset
            new_file = king_file + file_offset
            
            if 0 <= new_rank <= 7 and 0 <= new_file <= 7:
                square = chess.square(new_file, new_rank)
                king_zone.append(square)

    for square in king_zone:
        if board.is_attacked_by(board.turn, square):
            score += 20

    return score


def evaluate_piece_positions(board):
    """
    Apply bonus points for pieces in good positions (central positions are better)
    """
    pawn_table = [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ]

    knight_table = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ]

    score = 0

    # Evaluate white pawns
    for square in board.pieces(chess.PAWN, chess.WHITE):
        score += pawn_table[square]
    
    # Evaluate black pawns (flip table)
    for square in board.pieces(chess.PAWN, chess.BLACK):
        score -= pawn_table[63 - square]
    
    # Evaluate white knights
    for square in board.pieces(chess.KNIGHT, chess.WHITE):
        score += knight_table[square]
    
    # Evaluate black knights
    for square in board.pieces(chess.KNIGHT, chess.BLACK):
        score -= knight_table[63 - square]
    
    return score


def evaluate_king_saftey(board):
    """
    Evaluate king saftey and adjust score
    """
    score = 0

    if board.is_check():
        score -= 50

    return score
