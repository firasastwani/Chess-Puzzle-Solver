"""
Testing script for the chess puzzle solver.
Tests puzzles using either minimax or forward search.

Key difference:
- Minimax: Assumes adversarial opponent (tries to minimize our score)
- Forward Search: Uses known opponent responses from puzzle dataset (more appropriate for puzzles)

For puzzle solving where opponent moves are predetermined, forward search is more appropriate
since it doesn't waste time assuming the opponent will play optimally - it just uses the known responses.
"""

from datasets import load_dataset
import random
from main import solve_puzzle, filter_puzzles_by_mate_in_n
import time

def test_puzzles(num_puzzles=5, depth=6, use_engine_eval=True, filter_endgame=True, use_forward_search=False):
    """
    Load mateIn2 puzzles from the dataset and test the solver.
    
    Args:
        num_puzzles: Number of random puzzles to test
        depth: Search depth for minimax or max depth for forward search
        use_engine_eval: Whether to use engine evaluation
        filter_endgame: If True, only test endgame puzzles
        use_forward_search: If True, use forward search instead of minimax
    """
    print("="*70)
    print("LOADING LICHESS CHESS PUZZLES DATASET")
    print("="*70)
    
    # Load the dataset
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("Lichess/chess-puzzles")
    try:
        train_size = len(ds['train'])  # type: ignore
        print(f"Dataset loaded. Total puzzles: {train_size}")
    except TypeError:
        print("Dataset loaded (size unknown)")
    
    # Filter for mate in 2 puzzles
    print("\nFiltering for 'mateIn2' puzzles...")
    mate_in_2 = filter_puzzles_by_mate_in_n(ds['train'], 2)
    
    # Convert to list to get length and enable random sampling
    mate_in_2_list = list(mate_in_2)
    print(f"Found {len(mate_in_2_list)} mate in 2 puzzles")
    
    # Filter for endgame puzzles if requested
    if filter_endgame:
        print("\nFiltering for 'endgame' puzzles...")
        endgame_puzzles = [p for p in mate_in_2_list if 'endgame' in p.get('Themes', [])]
        print(f"Found {len(endgame_puzzles)} mate-in-2 endgame puzzles")
        puzzle_list = endgame_puzzles
    else:
        puzzle_list = mate_in_2_list
    
    if len(puzzle_list) == 0:
        theme_filter = "mateIn2 endgame" if filter_endgame else "mateIn2"
        print(f"ERROR: No {theme_filter} puzzles found in dataset!")
        return
    if len(puzzle_list) < num_puzzles:
        print(f"Warning: Only {len(puzzle_list)} puzzles available, testing all of them")
        selected_puzzles = puzzle_list
    else:
        selected_puzzles = random.sample(puzzle_list, num_puzzles)
    
    puzzle_type = "MATE-IN-2 ENDGAME" if filter_endgame else "MATE-IN-2"
    algorithm = "FORWARD SEARCH" if use_forward_search else "MINIMAX"
    print(f"\n{'='*70}")
    print(f"TESTING {len(selected_puzzles)} RANDOM {puzzle_type} PUZZLES")
    print(f"ALGORITHM: {algorithm}")
    print(f"{'='*70}\n")
    
    results = []
    total_start_time = time.time()
    
    for i, puzzle in enumerate(selected_puzzles, 1):
        print(f"\n{'#'*70}")
        print(f"# PUZZLE {i}/{len(selected_puzzles)}")
        print(f"{'#'*70}")
        print(f"Puzzle ID: {puzzle.get('PuzzleId', 'N/A')}")
        print(f"Rating: {puzzle.get('Rating', 'N/A')}")
        print(f"Themes: {puzzle.get('Themes', [])}")
        print(f"FEN: {puzzle.get('FEN', 'N/A')}")
        print(f"Expected Moves: {puzzle.get('Moves', 'N/A')}")
        print(f"{'#'*70}\n")
        
        puzzle_start_time = time.time()
        solved = solve_puzzle(puzzle, depth=depth, use_engine_eval=use_engine_eval, use_forward_search=use_forward_search)
        puzzle_time = time.time() - puzzle_start_time
        
        results.append({
            'puzzle_id': puzzle.get('PuzzleId', 'N/A'),
            'solved': solved,
            'time': puzzle_time,
            'rating': puzzle.get('Rating', 'N/A')
        })
        
        status = "✓ SOLVED" if solved else "✗ FAILED"
        print(f"\n{status} - Puzzle {i} completed in {puzzle_time:.2f}s")
        print(f"{'='*70}\n")
    
    total_time = time.time() - total_start_time
    
    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total puzzles tested: {len(results)}")
    print(f"Puzzles solved: {sum(1 for r in results if r['solved'])}")
    print(f"Puzzles failed: {sum(1 for r in results if not r['solved'])}")
    print(f"Success rate: {sum(1 for r in results if r['solved'])/len(results)*100:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per puzzle: {total_time/len(results):.2f}s")
    print(f"\nDetailed results:")
    for i, result in enumerate(results, 1):
        status = "✓" if result['solved'] else "✗"
        print(f"  {status} Puzzle {i} (ID: {result['puzzle_id']}, Rating: {result['rating']}): "
              f"{'SOLVED' if result['solved'] else 'FAILED'} in {result['time']:.2f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    random.seed(42)
    
    # Run tests
    # You can adjust depth and use_engine_eval based on your needs
    # Using depth=6 for better mate-in-2 puzzle solving
    
    print("\n" + "="*70)
    print("COMPARING MINIMAX vs FORWARD SEARCH")
    print("="*70)
    print("\nForward search is more appropriate for puzzles because:")
    print("  - Opponent moves are predetermined (not adversarial)")
    print("  - No need to assume opponent plays optimally")
    print("  - Just searches for sequences leading to checkmate")
    print("="*70 + "\n")
    
    # Test with minimax (original approach)
    print("\n" + "="*70)
    print("TEST 1: MINIMAX (assumes adversarial opponent)")
    print("="*70)
    test_puzzles(num_puzzles=5, depth=6, use_engine_eval=True, use_forward_search=False)
    
    # Test with forward search (more appropriate for puzzles)
    print("\n" + "="*70)
    print("TEST 2: FORWARD SEARCH (uses known opponent responses)")
    print("="*70)
    test_puzzles(num_puzzles=5, depth=6, use_engine_eval=True, use_forward_search=True)

