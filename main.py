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

def order_moves(board, moves, previous_move=None):
    """
    Order moves to improve alpha-beta pruning efficiency.
    Priority: Checkmates > Checks > Captures > Tactical Continuations > Other moves
    
    Args:
        board: chess.Board position
        moves: iterable of chess.Move objects
        previous_move: chess.Move from previous turn (for tactical continuity)
    
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
        
        # TACTICAL CONTINUATION BONUS: If we have a previous move, prioritize continuing the sequence
        if previous_move is not None:
            # Get the piece that would be on the destination square of previous move
            # (This works because we're looking for moves that continue with the same piece)
            move_piece = board.piece_at(move.from_square)
            
            # Bonus 1: Moving from the square where our previous move ended
            # This catches cases where the same piece continues (e.g., h2h8 -> h8h7)
            if move.from_square == previous_move.to_square:
                score += 50000  # Very large bonus - same piece continuing!
            
            # Bonus 2: Moving from the square where our previous move started
            # This catches cases where we use a different piece from the same square
            # (less common but still a continuation pattern)
            if move.from_square == previous_move.from_square:
                score += 20000  # Good bonus for same-square continuation
            
            # Bonus 3: Check if previous move was a check, and this move continues the pattern
            # We need to reconstruct the board state before opponent's response
            # For now, we'll use a simpler heuristic: if this move gives check and is from/to
            # squares related to the previous move, it's likely a continuation
            if board.gives_check(move):
                # If moving from where we just moved to, or to a related square
                if move.from_square == previous_move.to_square:
                    score += 40000  # Check continuation with same piece - very strong!
                elif move.from_square == previous_move.from_square:
                    score += 25000  # Check from same starting square
            
            # Bonus 4: Check sequences (check -> checkmate pattern)
            # If previous move gave check, prioritize moves that also give check
            # We can't easily check if previous move gave check without board history,
            # but we can prioritize check moves when we have a previous move context
            if board.gives_check(move) and move.from_square != previous_move.to_square:
                # This is a check from a different piece - might be coordination
                # Give a moderate bonus (less than same-piece continuation)
                score += 15000
        
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


# Global counter for nodes explored
nodes_explored = 0


def parse_puzzle_moves(puzzle_data):
    """
    Parse puzzle moves correctly.
    
    Puzzle format: The first move is the opponent's move (the blunder that creates the puzzle).
    Then moves alternate: our move, opponent response, our move, opponent response, etc.
    
    Example: 'e8f7 e2e6 f7f8 e6f7'
    - Index 0 (e8f7): Opponent's first move (applied to initial FEN)
    - Index 1 (e2e6): Our first move (to find)
    - Index 2 (f7f8): Opponent's response (from dataset)
    - Index 3 (e6f7): Our second move (to find)
    
    Args:
        puzzle_data: Dictionary containing puzzle information with 'FEN' and 'Moves' keys
    
    Returns:
        tuple: (board_after_opponent_first_move, our_expected_moves, opponent_responses)
            - board_after_opponent_first_move: chess.Board after opponent's first move
            - our_expected_moves: List of UCI strings for our moves (indices 1, 3, 5, ...)
            - opponent_responses: List of UCI strings for opponent responses (indices 2, 4, 6, ...)
    """
    board = load_puzzle_board(puzzle_data['FEN'])
    all_moves = puzzle_data['Moves'].split()
    
    # The first move (index 0) is the opponent's move - apply it to the board
    if len(all_moves) > 0:
        opponent_first_move = chess.Move.from_uci(all_moves[0])
        if opponent_first_move not in board.legal_moves:
            raise ValueError(f"Opponent's first move {all_moves[0]} is not legal in position {puzzle_data['FEN']}")
        board.push(opponent_first_move)
    
    # Our moves are at indices 1, 3, 5, etc. (odd indices after 0)
    # Opponent responses are at indices 2, 4, 6, etc. (even indices after 0)
    our_expected_moves = all_moves[1::2]  # Indices 1, 3, 5, 7...
    opponent_responses = all_moves[2::2]  # Indices 2, 4, 6, 8...
    
    return board, our_expected_moves, opponent_responses


def solve_puzzle(puzzle_data, stockfish_path=None, max_depth=10, use_engine_eval=False):
    """
    Solve puzzle using forward search with known opponent responses.
    This is more appropriate for puzzle solving where opponent moves are predetermined.
    
    Unlike minimax, this doesn't assume adversarial play - it just searches for sequences
    that lead to checkmate given the known opponent responses.
    
    Args:
        puzzle_data: Dictionary containing puzzle information
        stockfish_path: Path to Stockfish binary (None for auto-detect)
        max_depth: Maximum search depth (number of our moves to search)
        use_engine_eval: If True, use engine for position evaluation (currently not used in forward search)
    
    Returns:
        bool: True if puzzle was solved correctly
    """
    global nodes_explored
    nodes_explored = 0
    
    # Create engine if needed (though forward search doesn't use it much currently)
    engine = None
    if use_engine_eval:
        try:
            if stockfish_path is None:
                stockfish_paths = ["/opt/homebrew/bin/stockfish"]
                for path in stockfish_paths:
                    try:
                        engine = chess.engine.SimpleEngine.popen_uci(path)
                        break
                    except:
                        continue
                if engine is None:
                    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
            else:
                engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        except Exception as e:
            print(f"Warning: Could not initialize chess.engine: {e}")
            engine = None
    
    # Parse puzzle moves using the standard function
    board, our_expected_moves, opponent_responses = parse_puzzle_moves(puzzle_data)
    expected_num_our_moves = len(our_expected_moves)
    
    # Get the opponent's first move for display
    all_moves = puzzle_data['Moves'].split()
    if len(all_moves) > 0:
        print(f"Opponent plays: {all_moves[0]}")
    
    print(f"Solving puzzle {puzzle_data['PuzzleId']} using forward search")
    print(f"Full sequence: {' '.join(all_moves)}")
    print(f"Our expected moves: {' '.join(our_expected_moves)} ({expected_num_our_moves} moves to mate)")
    print(f"Opponent responses: {' '.join(opponent_responses) if opponent_responses else 'N/A'}")
    
    def forward_search(current_board, our_move_index, our_moves_so_far):
        """
        Recursive forward search that tries our moves and applies known opponent responses.
        
        Args:
            current_board: Current board position
            our_move_index: Which of our moves we're looking for (0-indexed)
            our_moves_so_far: List of our moves played so far
        
        Returns:
            tuple: (found_solution: bool, solution_moves: list)
        """
        global nodes_explored
        nodes_explored += 1
        
        # Base case: check if we've reached checkmate
        if current_board.is_checkmate():
            return True, our_moves_so_far
        
        # Base case: if we've made all expected moves, check if we're in checkmate
        if our_move_index >= expected_num_our_moves:
            return current_board.is_checkmate(), our_moves_so_far
        
        # Base case: if we've exceeded max depth, stop searching
        if our_move_index >= max_depth:
            return False, our_moves_so_far
        
        # Try all our legal moves
        ordered_moves = order_moves(current_board, current_board.legal_moves, 
                                   previous_move=our_moves_so_far[-1] if our_moves_so_far else None)
        
        for move in ordered_moves:
            # Make our move
            current_board.push(move)
            new_moves = our_moves_so_far + [move]
            
            # Check if this move immediately gives checkmate
            if current_board.is_checkmate():
                return True, new_moves
            
            # Apply opponent's known response (if available)
            if our_move_index < len(opponent_responses):
                opponent_response = chess.Move.from_uci(opponent_responses[our_move_index])
                
                # Verify the opponent response is legal (sanity check)
                if opponent_response in current_board.legal_moves:
                    current_board.push(opponent_response)
                    
                    # Recursively search for next move
                    found, solution = forward_search(current_board, our_move_index + 1, new_moves)
                    
                    # Undo opponent's move
                    current_board.pop()
                    
                    if found:
                        # Undo our move before returning
                        current_board.pop()
                        return True, solution
                else:
                    # Opponent response is not legal - this shouldn't happen in valid puzzles
                    print(f"  Warning: Expected opponent response {opponent_responses[our_move_index]} is not legal!")
            else:
                # No more opponent responses expected - check if we have forced mate
                # (all opponent moves lead to checkmate)
                all_lead_to_mate = True
                for opp_move in current_board.legal_moves:
                    current_board.push(opp_move)
                    if not current_board.is_checkmate():
                        all_lead_to_mate = False
                    current_board.pop()
                    if not all_lead_to_mate:
                        break
                
                if all_lead_to_mate and current_board.legal_moves:
                    # Undo our move before returning
                    current_board.pop()
                    return True, new_moves
            
            # Undo our move
            current_board.pop()
        
        # No solution found from this position
        return False, our_moves_so_far
    
    # Start the forward search
    start_time = time.time()
    found, solution_moves = forward_search(board, 0, [])
    elapsed = time.time() - start_time
    
    if found:
        print(f"\n✓ SOLUTION FOUND!")
        print(f"  Our moves: {' '.join([m.uci() for m in solution_moves])}")
        print(f"  Expected: {' '.join(our_expected_moves)}")
        print(f"  Nodes explored: {nodes_explored:,}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Nodes/sec: {nodes_explored/elapsed:,.0f}")
        
        # Verify solution matches expected
        solution_ucis = [m.uci() for m in solution_moves]
        if solution_ucis == our_expected_moves:
            print(f"  ✓ Matches expected solution exactly!")
        else:
            print(f"  → Different from expected, but leads to checkmate")
        
        solved = True
    else:
        print(f"\n✗ NO SOLUTION FOUND")
        print(f"  Nodes explored: {nodes_explored:,}")
        print(f"  Time: {elapsed:.2f}s")
        solved = False
    
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
    # Test the forward search puzzle solver
    print("\n" + "="*50)
    print("TESTING FORWARD SEARCH PUZZLE SOLVER")
    print("="*50)
    
    # Test 1: Simple puzzle
    print("\n" + "="*50)
    print("TEST 1: SIMPLE PUZZLE (Mate in 1)")
    print("="*50)
    simple_board = load_puzzle_board(simple_puzzle['FEN'])
    print("Position:")
    print(simple_board)
    print()
    
    result = solve_puzzle(simple_puzzle, max_depth=4, use_engine_eval=False)
    if result:
        print("\n✓ SUCCESS! Simple puzzle solved correctly")

    
    # Test 2: Harder puzzle
    print("\n" + "="*50)
    print("TEST 2: HARDER PUZZLE (Mate in 2)")
    print("="*50)
    print("This mate-in-2 puzzle requires finding a forced checkmate.")
    print("Testing with forward search...\n")
    
    result = solve_puzzle(example_puzzle, max_depth=4, use_engine_eval=False)
    
    if result:
        print(f"\n✓ SUCCESS! Puzzle solved correctly")
    else:
        print(f"\n✗ FAILED")
    
    # Uncomment to load and filter the full dataset:
    # print("\n" + "="*50)
    # print("Loading Lichess puzzles dataset...")
    # print("="*50)
    # ds = load_dataset("Lichess/chess-puzzles")
    # 
    # # Filter for mate in 2 puzzles
    # mate_in_2 = filter_puzzles_by_mate_in_n(ds['train'], 2)
    # print(f"\nFound {len(mate_in_2)} mate in 2 puzzles")
    