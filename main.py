from datasets import load_dataset
from stockfish import Stockfish
import chess
import chess.svg
from pathlib import Path
import time

def load_puzzle_board(fen_string):
    """
    Load a chess board from a FEN string.
    
    Args:
        fen_string: FEN notation string from the dataset
    Returns:
        chess.Board object
    """
    return chess.Board(fen_string)


def render_board_svg(board, last_move=None, check_square=None, size=400):
    """
    Renders a chess board as SVG to show position.

    Args: 
        board: chess.Board object
        last_move: chess.Move to highlight the last move 
        check_square: square in check for visibility
        size: size of the board output in pixesl

    Returns: 
        SVG string
    """
    return chess.svg.board(
        board,
        size = size, 
        lastmove = last_move, 
        check = check_square
    )


def save_board_svg(board, filename, last_move=None, check_square=None, size=400):
    """
    Save a chess board as an SVG file

    Args: 
        board: chess.Board object
        filename: output filename (.svg file)
        last_move: chess.Move object 
        check_square: Square in check
        size: Size of the board in pixels
    """

    svg = render_board_svg(board, last_move, check_square, size)
    Path(filename).write_text(svg)
    print(f"Saved board to {filename}")


def render_puzzle_sequence(puzzle_data, output_dir="puzzle_renders", size=400):
    """
    Render all positions in a puzzle as seperate svg files

    Args: 
        puzzle_data: Dict containing puzzle info
        output_dir: Directory to save svg outputs to
        size: Size of each board in pixels

    Returns: 
        List of generated filenames
    """

    # create the dir if it dosent exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    board = load_puzzle_board(puzzle_data['FEN'])
    move_string = puzzle_data['Moves']
    move_list = move_string.split()

    puzzle_id = puzzle_data['PuzzleId']
    filenames = []

    initial_position = output_path / f"{puzzle_id}_00_initial.svg"
    save_board_svg(board, initial_position, size=size)
    filenames.append(str(initial_position))

    for i, move_uci in enumerate(move_list, 1):
        move = chess.Move.from_uci(move_uci)

        board.push(move)

        check_square = board.king(board.turn) if board.is_check() else None

        move_file = output_path / f"{puzzle_id}_{i:02d}_{move_uci}.svg"
        save_board_svg(board, move_file, last_move=move, check_square=check_square, size=size)
        filenames.append(str(move_file))

    return filenames


def apply_puzzle_moves(board, moves_string):
    """
    Apply moves to the board. Moves are in UCI format (e.g., 'e2e4').
    
    Args:
        board: chess.Board object
        moves_string: Space-separated UCI moves (e.g., 'e8f7 e2e6 f7f8 e6f7')
    Returns:
        List of board states after each move
    """
    moves = moves_string.split()
    boards = [board.copy()]
    
    for move_uci in moves:
        move = chess.Move.from_uci(move_uci)
        board.push(move)
        boards.append(board.copy())
    
    return boards


def get_puzzle_solution(puzzle_data):
    """
    Extract the full solution for a puzzle.
    
    Args:
        puzzle_data: Dictionary containing puzzle information
    Returns:
        Tuple of (initial_board, solution_boards, is_checkmate)
    """
    board = load_puzzle_board(puzzle_data['FEN'])
    solution_boards = apply_puzzle_moves(board.copy(), puzzle_data['Moves'])
    final_board = solution_boards[-1]
    is_checkmate = final_board.is_checkmate()
    
    return board, solution_boards, is_checkmate


def filter_puzzles_by_mate_in_n(dataset, n):
    """
    Filter puzzles that are mate in N moves.
    
    Args:
        dataset: HuggingFace dataset
        n: Number of moves to mate
    Returns:
        Filtered dataset
    """
    theme_to_find = f'mateIn{n}'
    return dataset.filter(lambda x: theme_to_find in x['Themes'])


# custom evaluation function
def evaluate_position(board):
    """
    Static evaluation function for minimax algo.
    Returns a score from the perspective of the current player. 

    Positive = good for white, Negative = good for black

    Returns: 
        int: Position score (in centipawns) 100 = 1 pawn advantage
    """

    # worst case for evaluated side (white)
    if board.is_checkmate():
        return -20000
    
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

    # calculate material counts
    for piece_type in piece_values: 
        white_pieces = len(board.pieces(piece_type, chess.WHITE))
        black_pieces = len(board.pieces(piece_type, chess.BLACK))

        score += (white_pieces - black_pieces) * piece_values[piece_type]
    
    # Add positional bonuses
    score += evaluate_piece_positions(board)
    
    # Add king safety evaluation
    score += evaluate_king_saftey(board)
    
    # Add mobility (number of legal moves is good)
    score += len(list(board.legal_moves)) * 10
    
    # Return from white's perspective
    return score if board.turn == chess.WHITE else -score


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


def order_moves(board, moves):
    """
    Order moves to improve alpha-beta pruning efficiency.
    Priority: Checkmates > Checks > Captures > Other moves
    
    Args:
        board: chess.Board position
        moves: iterable of chess.Move objects
    
    Returns:
        list of moves sorted by priority
    """
    scored_moves = []
    
    for move in moves:
        score = 0
        
        # Prioritize checks (most important for mate puzzles)
        board.push(move)
        if board.is_checkmate():
            score += 100000  # Checkmate is best
        elif board.is_check():
            score += 10000  # Checks are very good
        board.pop()
        
        # Prioritize captures
        if board.is_capture(move):
            # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
            captured = board.piece_at(move.to_square)
            if captured:
                victim_value = {
                    chess.PAWN: 100,
                    chess.KNIGHT: 300,
                    chess.BISHOP: 300,
                    chess.ROOK: 500,
                    chess.QUEEN: 900,
                    chess.KING: 0
                }.get(captured.piece_type, 0)
                score += victim_value
        
        # Prioritize center moves
        to_rank = chess.square_rank(move.to_square)
        to_file = chess.square_file(move.to_square)
        if 2 <= to_rank <= 5 and 2 <= to_file <= 5:
            score += 10
        
        scored_moves.append((score, move))
    
    # Sort by score (highest first)
    scored_moves.sort(key=lambda x: x[0], reverse=True)
    
    return [move for score, move in scored_moves]


def minimax(board, depth, alpha, beta, maximizing):
    """
    Minimax search with alpha-beta pruning and move ordering.

    Args: 
        board: chess.Board current position
        depth: remaining search depth
        alpha: best score for maximizing player
        beta: best score for minimizing player
        maximizing: boolean set to True if maximizing players turn

    Returns: 
        tuple: (best_score, best_move)
    """
    global nodes_explored
    nodes_explored += 1
    
    if depth == 0 or board.is_game_over():
        return evaluate_position(board), None
    
    best_move = None
    
    # Order moves for better pruning
    ordered_moves = order_moves(board, board.legal_moves)

    if maximizing:
        max_eval = float('-inf')
        for move in ordered_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff
        
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in ordered_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff
        
        return min_eval, best_move
    

# Global counter for nodes explored
nodes_explored = 0


def solve_puzzle(puzzle_data, depth=6):
    """
    Attempt to solve the chess puzzle using minimax.
    """
    global nodes_explored
    
    board = load_puzzle_board(puzzle_data['FEN'])
    expected_moves = puzzle_data['Moves'].split()
    
    print(f"Solving puzzle {puzzle_data['PuzzleId']}")
    print(f"Expected solution: {' '.join(expected_moves)}")
    
    solved = True
    total_nodes = 0
    total_time = 0
    
    for i, expected_move in enumerate(expected_moves):
        nodes_explored = 0
        start_time = time.time()
        
        score, best_move = minimax(board, depth, float('-inf'), float('inf'), board.turn)
        
        elapsed = time.time() - start_time
        total_nodes += nodes_explored
        total_time += elapsed
        
        if best_move is None:
            print(f"Move {i+1}: No move found (depth too shallow or game over)")
            solved = False
            break
        
        if best_move.uci() != expected_move:
            print(f"Move {i+1}: Found {best_move.uci()}, expected {expected_move}")
            print(f"  Nodes: {nodes_explored:,} | Time: {elapsed:.2f}s")
            solved = False
            break
        else:
            print(f"Move {i+1}: {best_move.uci()} (score: {score})")
            print(f"  Nodes: {nodes_explored:,} | Time: {elapsed:.2f}s | Nodes/sec: {nodes_explored/elapsed:,.0f}")
        
        board.push(best_move)
    
    if solved:
        print(f"\nTotal - Nodes: {total_nodes:,} | Time: {total_time:.2f}s")
    
    return solved





# Example usage with your puzzle data:
if __name__ == "__main__":
    # Example puzzle data
    example_puzzle = {
        'PuzzleId': '000hf',
        'GameId': '71ygsFeE/black#38',
        'FEN': 'r1bqk2r/pp1nbNp1/2p1p2p/8/2BP4/1PN3P1/P3QP1P/3R1RK1 b kq - 0 19',
        'Moves': 'e8f7 e2e6 f7f8 e6f7',
        'Rating': 1575,
        'RatingDeviation': 75,
        'Popularity': 92,
        'NbPlays': 674,
        'Themes': ['mate', 'mateIn2', 'middlegame', 'short'],
        'OpeningTags': ['Horwitz_Defense', 'Horwitz_Defense_Other_variations']
    }
    
    # Load the board from FEN
    board = load_puzzle_board(example_puzzle['FEN'])
    print("Initial board position:")
    print(board)
    print("\nFEN:", board.fen())
    print("\nIt's", "White's" if board.turn == chess.WHITE else "Black's", "turn")
    print(f"\nPuzzle Themes: {example_puzzle['Themes']}")
    
    # Apply the puzzle solution moves
    print("\n" + "="*50)
    print("Applying puzzle solution moves:")
    print("="*50)
    
    boards = apply_puzzle_moves(board.copy(), example_puzzle['Moves'])
    move_list = example_puzzle['Moves'].split()
    
    for i, (move_uci, b) in enumerate(zip(move_list, boards[1:]), 1):
        print(f"\nMove {i}: {move_uci}")
        print(b)
        if b.is_checkmate():
            print("CHECKMATE!")

    # Render the puzzle as SVG files
    print("\n" + "="*50)
    print("Rendering puzzle as SVG files:")
    print("="*50)
    rendered_files = render_puzzle_sequence(example_puzzle, output_dir="puzzle_renders")
    print(f"\nGenerated {len(rendered_files)} SVG files:")
    for filename in rendered_files:
        print(f"  - {filename}")
    
    # You can also render individual positions
    print("\n" + "="*50)
    print("Rendering individual position:")
    print("="*50)
    final_board = boards[-1]
    save_board_svg(
        final_board, 
        "checkmate_position.svg",
        last_move=chess.Move.from_uci(move_list[-1]),
        check_square=final_board.king(final_board.turn),
        size=600
    )
    
    # Test the minimax solver
    print("\n" + "="*50)
    print("TESTING MINIMAX PUZZLE SOLVER")
    print("="*50)
    print("\nThis mate-in-2 puzzle requires finding a forced checkmate.")
    print("Testing different search depths...\n")
    
    # Test with different depths
    for depth in [4, 6]:
        print(f"\n{'='*50}")
        print(f"Testing with depth={depth}")
        print('='*50)
        
        result = solve_puzzle(example_puzzle, depth=depth)
        
        if result:
            print(f"\n✓ SUCCESS! Puzzle solved correctly at depth {depth}")
        else:
            print(f"\n✗ FAILED at depth {depth}")
        
        # Only try deeper if shallow failed
        if result:
            break
    
    # Uncomment to load and filter the full dataset:
    # print("\n" + "="*50)
    # print("Loading Lichess puzzles dataset...")
    # print("="*50)
    # ds = load_dataset("Lichess/chess-puzzles")
    # 
    # # Filter for mate in 2 puzzles
    # mate_in_2 = filter_puzzles_by_mate_in_n(ds['train'], 2)
    # print(f"\nFound {len(mate_in_2)} mate in 2 puzzles")
