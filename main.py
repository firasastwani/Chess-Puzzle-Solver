from datasets import load_dataset
from stockfish import Stockfish
import chess
import chess.engine
import chess.svg
from pathlib import Path
import time
import sys
from custom_eval import (
    evaluate_position_fast,
    evaluate_position_engine,
    evaluate_king_attacks,
    evaluate_piece_positions,
    evaluate_king_saftey
)

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
            score += 1000000  # Checkmate is best
        elif board.is_check():
            score += 100000  # Checks are very good
        board.pop()
        
        # Prioritize captures HEAVILY (especially for hanging pieces!)
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
                # Add huge bonus for captures (10000 + piece value)
                score += 10000 + victim_value
        
        # Prioritize center moves
        to_rank = chess.square_rank(move.to_square)
        to_file = chess.square_file(move.to_square)
        if 2 <= to_rank <= 5 and 2 <= to_file <= 5:
            score += 10
        
        scored_moves.append((score, move))
    
    # Sort by score (highest first)
    scored_moves.sort(key=lambda x: x[0], reverse=True)
    
    return [move for score, move in scored_moves]


def quiescence_search_with_engine(board, alpha, beta, engine, max_depth=4):
    """
    Quiescence search using engine for evaluations.
    Continues searching captures and checks to find forced mates.
    
    Args:
        board: chess.Board position
        alpha: alpha value
        beta: beta value
        engine: chess.engine.SimpleEngine instance
        max_depth: maximum quiescence depth
    
    Returns:
        int: evaluation score
    """
    global nodes_explored
    nodes_explored += 1
    
    # Check for game over
    if board.is_checkmate():
        return -100000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    # Stand-pat score using engine
    # Use slightly higher depth in quiescence since we're in tactical sequences
    stand_pat = evaluate_position_engine(board, engine, depth_limit=2, time_limit=0.02)
    
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
    
    if max_depth <= 0:
        return stand_pat
    
    # If in check, must search ALL moves
    if board.is_check():
        for move in order_moves(board, board.legal_moves):
            board.push(move)
            score = -quiescence_search_with_engine(board, -beta, -alpha, engine, max_depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
    else:
        # Only search captures and checks (forcing moves)
        for move in order_moves(board, board.legal_moves):
            if not (board.is_capture(move) or board.gives_check(move)):
                continue
                
            board.push(move)
            score = -quiescence_search_with_engine(board, -beta, -alpha, engine, max_depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
    
    return alpha


def quiescence_search(board, alpha, beta, max_depth=4):
    """
    Quiescence search - continue searching captures and checks after depth limit.
    This helps find tactical sequences like forced mates.
    Uses fast material evaluation (Stockfish is too slow!).
    
    Args:
        board: chess.Board position
        alpha: alpha value
        beta: beta value
        max_depth: maximum quiescence depth to prevent infinite search
    
    Returns:
        int: evaluation score
    """
    global nodes_explored
    nodes_explored += 1
    
    # Check for game over
    if board.is_checkmate():
        return -100000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    # Stand-pat score (can we just stop here?)
    stand_pat = evaluate_position_fast(board)
    
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
    
    if max_depth <= 0:
        return stand_pat
    
    # If in check, must search ALL moves (you can't ignore check!)
    if board.is_check():
        for move in order_moves(board, board.legal_moves):
            board.push(move)
            score = -quiescence_search(board, -beta, -alpha, max_depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
    else:
        # Only search captures and checks (forcing moves) when not in check
        for move in order_moves(board, board.legal_moves):
            # Only tactical moves
            if not (board.is_capture(move) or board.gives_check(move)):
                continue
                
            board.push(move)
            score = -quiescence_search(board, -beta, -alpha, max_depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
    
    return alpha


def minimax(board, depth, alpha, beta, maximizing, engine=None, ply_from_root=0, use_engine=False):
    """
    Minimax search with alpha-beta pruning and move ordering.
    Can use chess.engine for evaluation (faster than stockfish package).
    
    Args:
        board: chess.Board current position
        depth: remaining search depth
        alpha: best score for maximizing player
        beta: best score for minimizing player
        maximizing: boolean set to True if maximizing players turn
        engine: chess.engine.SimpleEngine instance (optional)
        ply_from_root: distance from root (for preferring faster mates)
        use_engine: If True, use engine at depth 0; else use quiescence search
    
    Returns:
        tuple: (best_score, best_move)
    """
    global nodes_explored
    nodes_explored += 1
    
    if board.is_checkmate():
        # Prefer checkmates closer to root (faster mates)
        return -100000 + ply_from_root, None
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0, None
    
    # At depth 0, use engine evaluation directly (as it was working at depth 4)
    if depth == 0:
        if use_engine and engine:
            # Use chess.engine with very low depth for speed (this was working!)
            return evaluate_position_engine(board, engine, depth_limit=1, time_limit=0.01), None
        else:
            # Use quiescence search to extend tactical sequences
            return quiescence_search(board, alpha, beta, max_depth=5), None
    
    best_move = None
    
    # Order moves for better pruning
    ordered_moves = order_moves(board, board.legal_moves)
    
    if maximizing: 
        max_eval = float('-inf')
        for move in ordered_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, False, engine, ply_from_root + 1, use_engine)
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
            eval_score, _ = minimax(board, depth - 1, alpha, beta, True, engine, ply_from_root + 1, use_engine)
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


def solve_puzzle(puzzle_data, stockfish_path=None, depth=6, use_engine_eval=False):
    """
    Attempt to solve the chess puzzle using minimax.
    
    Args:
        puzzle_data: Dictionary containing puzzle information
        stockfish_path: Path to Stockfish binary (None for auto-detect)
        depth: Search depth
        use_engine_eval: If True, use chess.engine for evaluation at depth 0 (faster than stockfish package)
    
    Returns:
        bool: True if puzzle was solved correctly
    """
    global nodes_explored
    
    # Create chess.engine instance if requested
    engine = None
    if use_engine_eval:
        try:
            # Try to find Stockfish binary
            if stockfish_path is None:
                # Try common paths
                stockfish_paths = [
                    "/opt/homebrew/bin/stockfish",
                ]
                for path in stockfish_paths:
                    try:
                        engine = chess.engine.SimpleEngine.popen_uci(path)
                        print(f"Using chess.engine with Stockfish at: {path}")
                        break
                    except:
                        continue
                if engine is None:
                    # Try auto-detect
                    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
                    print("Using chess.engine with auto-detected Stockfish")
            else:
                engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                print(f"Using chess.engine with Stockfish at: {stockfish_path}")
        except Exception as e:
            print(f"Warning: Could not initialize chess.engine: {e}")
            print("Falling back to material evaluation...")
            use_engine_eval = False
            engine = None
    
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
        
        score, best_move = minimax(board, depth, float('-inf'), float('inf'), board.turn, engine, ply_from_root=0, use_engine=use_engine_eval)
        
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
        else:
            print(f"Move {i+1}: {best_move.uci()} (score: {score})")
            print(f"  Nodes: {nodes_explored:,} | Time: {elapsed:.2f}s | Nodes/sec: {nodes_explored/elapsed:,.0f}")
        
        board.push(best_move)
    
    if solved:
        print(f"\nTotal - Nodes: {total_nodes:,} | Time: {total_time:.2f}s")
    
    # Clean up engine
    if engine:
        try:
            engine.quit()
        except:
            pass
    
    return solved





# Example usage with your puzzle data:
if __name__ == "__main__":
    # Simple test puzzle - Mate in 1
    simple_puzzle = {
        'PuzzleId': 'simple_test',
        'FEN': '4k3/p1R5/5R2/4P3/2P5/4n1P1/4r2P/7K w - - 3 35',  
        'Moves': 'f6h6 e2e1', 
        'Themes': ['mate', 'mateIn1'],
    }
    
    # Example puzzle data - Mate in 2 (harder)
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
    # Test the minimax solver
    print("\n" + "="*50)
    print("TESTING MINIMAX PUZZLE SOLVER")
    print("="*50)
    
    # Try chess.engine evaluation (faster than stockfish package)
    print("Testing with chess.engine evaluation (python-chess engine interface)")
    print("This uses Stockfish via chess.engine with very low depth for speed")
    
    # Test 1: Simple puzzle
    print("\n" + "="*50)
    print("TEST 1: SIMPLE PUZZLE (Mate in 1)")
    print("="*50)
    simple_board = load_puzzle_board(simple_puzzle['FEN'])
    print("Position:")
    print(simple_board)
    print()
    
    # Test with engine evaluation
    # For mate-in-1, we need at least depth 2 (opponent move + our response)
    # But we're using depth 5 to be safe
    result = solve_puzzle(simple_puzzle, depth=4, use_engine_eval=True)
    if result:
        print("\n✓ SUCCESS! Simple puzzle solved correctly")
    else:
        print("\n✗ FAILED simple puzzle")
        print("Trying without engine (material eval only)...")
        result = solve_puzzle(simple_puzzle, depth=5, use_engine_eval=False)
    
    # Test 2: Harder puzzle
    print("\n" + "="*50)
    print("TEST 2: HARDER PUZZLE (Mate in 2)")
    print("="*50)
    print("This mate-in-2 puzzle requires finding a forced checkmate.")
    print("Testing with chess.engine evaluation...\n")
    
    # Test with different depths
    for depth in [3, 5]:
        print(f"\n{'='*50}")
        print(f"Testing with depth={depth}")
        print('='*50)
        
        result = solve_puzzle(example_puzzle, depth=depth, use_engine_eval=True)
        
        if result:
            print(f"\n✓ SUCCESS! Puzzle solved correctly at depth {depth}")
        else:
            print(f"\n✗ FAILED at depth {depth}")
        
        # Only try deeper if shallow failed
        if result:
            break
        
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
    