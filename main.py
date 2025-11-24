from datasets import load_dataset
from stockfish import Stockfish
import chess
import chess.svg
from pathlib import Path

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
    
    # Uncomment to load and filter the full dataset:
    # print("\n" + "="*50)
    # print("Loading Lichess puzzles dataset...")
    # print("="*50)
    # ds = load_dataset("Lichess/chess-puzzles")
    # 
    # # Filter for mate in 2 puzzles
    # mate_in_2 = filter_puzzles_by_mate_in_n(ds['train'], 2)
    # print(f"\nFound {len(mate_in_2)} mate in 2 puzzles")

    