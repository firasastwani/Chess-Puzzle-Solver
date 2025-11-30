"""
Comprehensive test file for puzzle solving with different evaluation methods.

Tests:
1. Custom evaluators with mateIn2 puzzles (5 puzzles)
2. Stockfish evaluator with mateIn2 puzzles (5 puzzles, same as test 1)
3. Custom evaluators with random puzzles (5 puzzles)
4. Stockfish evaluator with random puzzles (5 puzzles, same as test 3)
"""

from datasets import load_dataset
import random
import chess
import chess.engine
import time
from pathlib import Path
from main import filter_puzzles_by_mate_in_n, order_moves, parse_puzzle_moves, save_board_svg, render_board_svg
from custom_eval import (
    evaluate_position_fast,
)
import chess.engine

# Global counter for nodes explored
nodes_explored = 0


def solve_puzzle_with_eval(puzzle_data, max_depth=10, use_engine_eval=False, stockfish_path=None):
    """
    Solve puzzle using forward search with evaluation functions.
    
    Args:
        puzzle_data: Dictionary containing puzzle information
        max_depth: Maximum search depth (number of our moves to search)
        use_engine_eval: If True, use Stockfish engine; False uses custom fast evaluation
        stockfish_path: Path to Stockfish binary (None for auto-detect)
    
    Returns:
        bool: True if puzzle was solved correctly
    """
    global nodes_explored
    nodes_explored = 0
    
    # Initialize engine ONLY if using engine evaluation
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
            print("Falling back to custom evaluation...")
            use_engine_eval = False
            engine = None
    else:
        # For custom evaluation, ensure no engine is initialized
        engine = None
    
    # Parse puzzle moves using the standard function
    board, our_expected_moves, opponent_responses = parse_puzzle_moves(puzzle_data)
    expected_num_our_moves = len(our_expected_moves)
    
    # Create output directory for puzzle renders
    puzzle_id = puzzle_data.get('PuzzleId', 'unknown')
    output_dir = Path("puzzle_renders") / f"{puzzle_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save initial position (after opponent's first move)
    all_moves = puzzle_data['Moves'].split()
    if len(all_moves) > 0:
        print(f"Opponent plays: {all_moves[0]}")
        opponent_first_move = chess.Move.from_uci(all_moves[0])
        check_square = board.king(board.turn) if board.is_check() else None
        save_board_svg(
            board, 
            output_dir / "00_initial.svg",
            last_move=opponent_first_move,
            check_square=check_square
        )
    
    eval_name = "STOCKFISH" if use_engine_eval else "CUSTOM"
    print(f"Solving puzzle {puzzle_id} using forward search with {eval_name} evaluation")
    print(f"Full sequence: {' '.join(all_moves)}")
    print(f"Our expected moves: {' '.join(our_expected_moves)} ({expected_num_our_moves} moves to mate)")
    print(f"Opponent responses: {' '.join(opponent_responses) if opponent_responses else 'N/A'}")
    
    def evaluate_position_custom(board):
        """
        Evaluation function wrapper.
        - If use_engine_eval=True: Uses Stockfish engine (external)
        - If use_engine_eval=False: Uses ONLY functions from custom_eval.py
        """
        if use_engine_eval and engine:
            # Use Stockfish engine for evaluation (external, not from custom_eval.py)
            try:
                info = engine.analyse(
                    board,
                    chess.engine.Limit(depth=2, time=0.02)
                )
                # Get score safely
                score_obj = info.get("score")
                if score_obj is None:
                    return evaluate_position_fast(board)
                
                score = score_obj.relative.score(mate_score=100000)
                
                # Handle None (mate) or convert to int
                if score is None:
                    if score_obj.relative.is_mate():
                        mate_in = score_obj.relative.mate()
                        score = (50000 - abs(mate_in) * 100) if mate_in else 50000
                    else:
                        score = 0
                
                score_int = int(score)
                if board.turn == chess.BLACK:
                    score_int = -score_int
                
                return score_int
            except Exception:
                # Fallback to custom evaluation if engine fails
                return evaluate_position_fast(board)
        else:
            # CUSTOM EVALUATION: Only uses functions from custom_eval.py
            # Currently using evaluate_position_fast, but can be changed to:
            # - evaluate_position() for full custom evaluation
            # - evaluate_position_fast() for fast material-based evaluation
            return evaluate_position_fast(board)
    
    def forward_search_with_eval(current_board, our_move_index, our_moves_so_far, alpha=float('-inf')):
        """
        Recursive forward search with evaluation and alpha-beta pruning.
        Uses only dataset opponent responses - drops paths if response is illegal.
        
        Args:
            current_board: Current board position
            our_move_index: Which of our moves we're looking for (0-indexed)
            our_moves_so_far: List of our moves played so far
            alpha: Best score found so far (for pruning)
        
        Returns:
            tuple: (found_solution: bool, solution_moves: list, best_score: float)
        """
        global nodes_explored
        nodes_explored += 1
        
        # Base case: check if we've reached checkmate
        if current_board.is_checkmate():
            return True, our_moves_so_far, 100000  # High score for checkmate
        
        # Base case: if we've made all expected moves, check if we're in checkmate
        if our_move_index >= expected_num_our_moves:
            if current_board.is_checkmate():
                return True, our_moves_so_far, 100000
            return False, our_moves_so_far, evaluate_position_custom(current_board)
        
        # Base case: if we've exceeded max depth, stop searching
        if our_move_index >= max_depth:
            return False, our_moves_so_far, evaluate_position_custom(current_board)
        
        # Evaluate current position
        current_eval = evaluate_position_custom(current_board)
        
        # Try all our legal moves, ordered by evaluation
        # Step 1: Initial heuristic ordering (pure heuristics, no evaluation function)
        # This uses order_moves() which prioritizes: checkmates > checks > captures > tactical continuations
        ordered_moves = order_moves(current_board, current_board.legal_moves, 
                                   previous_move=our_moves_so_far[-1] if our_moves_so_far else None)
        
        # Step 2: Additional sorting by position evaluation
        # When use_engine_eval=False: Uses evaluate_position_fast() from custom_eval.py (pure custom)
        # When use_engine_eval=True: Uses Stockfish engine evaluation
        move_scores = []
        for move in ordered_moves:
            current_board.push(move)
            move_eval = evaluate_position_custom(current_board)  # Uses custom or engine based on use_engine_eval
            current_board.pop()
            move_scores.append((move_eval, move))
        
        # Sort by evaluation (higher is better for us)
        move_scores.sort(key=lambda x: x[0], reverse=True)
        ordered_moves = [move for _, move in move_scores]
        
        best_score = float('-inf')
        best_solution = None
        found_checkmate = False
        
        # Explore ALL moves to find the optimal one based on evaluation
        for move in ordered_moves:
            # Make our move
            current_board.push(move)
            new_moves = our_moves_so_far + [move]
            
            # Check if this move immediately gives checkmate
            if current_board.is_checkmate():
                # Found checkmate, but continue exploring to see if there's a better one
                # (though checkmate is always best, we still want to count all nodes)
                found_checkmate = True
                score = 100000
                if score > best_score:
                    best_score = score
                    best_solution = new_moves
                current_board.pop()
                # Continue exploring other moves to count all nodes
                continue
            
            # Apply opponent's known response (if available)
            if our_move_index < len(opponent_responses):
                opponent_response = chess.Move.from_uci(opponent_responses[our_move_index])
                
                # Verify the opponent response is legal
                if opponent_response in current_board.legal_moves:
                    current_board.push(opponent_response)
                    
                    # Recursively search for next move
                    found, solution, score = forward_search_with_eval(
                        current_board, our_move_index + 1, new_moves, alpha
                    )
                    
                    # Undo opponent's move
                    current_board.pop()
                    
                    if found:
                        # Found checkmate sequence, but continue exploring to find optimal
                        found_checkmate = True
                        if score > best_score:
                            best_score = score
                            best_solution = solution
                        # Continue exploring other moves
                        current_board.pop()
                        continue
                    
                    # Update best score and alpha
                    if score > best_score:
                        best_score = score
                        best_solution = solution
                    
                    # Alpha-beta pruning: if this branch is worse than what we've seen, skip it
                    if score < alpha:
                        current_board.pop()
                        continue  # Prune this branch
                    
                    alpha = max(alpha, score)
                else:
                    # Opponent response is not legal - drop this path
                    # Don't try all opponent moves, just continue to next of our moves
                    pass
            else:
                # No more opponent responses expected - check if we have forced mate
                all_lead_to_mate = True
                for opp_move in current_board.legal_moves:
                    current_board.push(opp_move)
                    if not current_board.is_checkmate():
                        all_lead_to_mate = False
                    current_board.pop()
                    if not all_lead_to_mate:
                        break
                
                if all_lead_to_mate and current_board.legal_moves:
                    found_checkmate = True
                    score = 100000
                    if score > best_score:
                        best_score = score
                        best_solution = new_moves
                    current_board.pop()
                    continue
            
            # Undo our move
            current_board.pop()
        
        # Return the best solution found (checkmate if found, otherwise best evaluation)
        if found_checkmate and best_solution is not None:
            return True, best_solution, best_score
        else:
            return False, best_solution if best_solution else our_moves_so_far, best_score
    
    # Start the forward search
    start_time = time.time()
    found, solution_moves, final_score = forward_search_with_eval(board, 0, [])
    elapsed = time.time() - start_time
    
    if found and solution_moves is not None:
        print(f"\n✓ SOLUTION FOUND!")
        print(f"  Our moves: {' '.join([m.uci() for m in solution_moves])}")
        print(f"  Expected: {' '.join(our_expected_moves)}")
        print(f"  Nodes explored: {nodes_explored:,}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Nodes/sec: {nodes_explored/elapsed:,.0f}" if elapsed > 0 else "  Nodes/sec: N/A")
        
        # Verify solution matches expected
        solution_ucis = [m.uci() for m in solution_moves]
        if solution_ucis == our_expected_moves:
            print(f"  ✓ Matches expected solution exactly!")
        else:
            print(f"  → Different from expected, but leads to checkmate")
        
        # Visualize the solution sequence in correct order
        visualize_solution = board.copy()
        move_number = 1  # Start from 1 (0 is initial position)
        
        for i, move in enumerate(solution_moves):
            # Our move
            visualize_solution.push(move)
            check_square = visualize_solution.king(visualize_solution.turn) if visualize_solution.is_check() else None
            save_board_svg(
                visualize_solution,
                output_dir / f"{move_number:02d}_{move.uci()}.svg",
                last_move=move,
                check_square=check_square
            )
            move_number += 1
            
            # Apply opponent response if available
            if i < len(opponent_responses):
                opp_move = chess.Move.from_uci(opponent_responses[i])
                if opp_move in visualize_solution.legal_moves:
                    visualize_solution.push(opp_move)
                    check_square = visualize_solution.king(visualize_solution.turn) if visualize_solution.is_check() else None
                    save_board_svg(
                        visualize_solution,
                        output_dir / f"{move_number:02d}_{opp_move.uci()}.svg",
                        last_move=opp_move,
                        check_square=check_square
                    )
                    move_number += 1
        
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


def test_puzzles(puzzles, test_name, use_engine_eval=False, max_depth=10):
    """
    Test a set of puzzles with the specified evaluation method.
    
    Args:
        puzzles: List of puzzle dictionaries
        test_name: Name of the test (for display)
        use_engine_eval: If True, use Stockfish; False uses custom evaluation
        max_depth: Maximum search depth
    """
    print(f"\n{'='*70}")
    print(f"{test_name}")
    print(f"{'='*70}\n")
    
    results = []
    total_start_time = time.time()
    
    for i, puzzle in enumerate(puzzles, 1):
        print(f"\n{'#'*70}")
        print(f"# PUZZLE {i}/{len(puzzles)}")
        print(f"{'#'*70}")
        print(f"Puzzle ID: {puzzle.get('PuzzleId', 'N/A')}")
        print(f"Rating: {puzzle.get('Rating', 'N/A')}")
        print(f"Themes: {puzzle.get('Themes', [])}")
        print(f"FEN: {puzzle.get('FEN', 'N/A')}")
        print(f"Expected Moves: {puzzle.get('Moves', 'N/A')}")
        print(f"{'#'*70}\n")
        
        puzzle_start_time = time.time()
        solved = solve_puzzle_with_eval(
            puzzle, 
            max_depth=max_depth, 
            use_engine_eval=use_engine_eval
        )
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
    
    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    print("="*70)
    print("COMPREHENSIVE PUZZLE SOLVING TESTS")
    print("="*70)
    print("\nThis test suite compares:")
    print("  - Custom evaluation (fast material-based)")
    print("  - Stockfish engine evaluation")
    print("  - On both mate-in-2 puzzles and random mate puzzles")
    print("="*70)
    
    # Load dataset
    print("\nLoading dataset from HuggingFace...")
    ds = load_dataset("Lichess/chess-puzzles")
    try:
        train_size = len(ds['train'])  # type: ignore
        print(f"Dataset loaded. Total puzzles: {train_size}")
    except TypeError:
        print("Dataset loaded (size unknown)")
    
    # ========================================================================
    # TEST 1 & 2: Mate-in-2 puzzles (same puzzles for both tests)
    # ========================================================================
    print("\n" + "="*70)
    print("PREPARING MATE-IN-2 PUZZLES")
    print("="*70)
    
    print("\nFiltering for 'mateIn2' puzzles...")
    mate_in_2 = filter_puzzles_by_mate_in_n(ds['train'], 2)
    mate_in_2_list = list(mate_in_2)
    print(f"Found {len(mate_in_2_list)} mate-in-2 puzzles")
    
    # Select 5 random mate-in-2 puzzles
    if len(mate_in_2_list) < 5:
        mate2_puzzles = mate_in_2_list
    else:
        mate2_puzzles = random.sample(mate_in_2_list, 5)
    
    print(f"Selected {len(mate2_puzzles)} puzzles for tests 1 & 2\n")
    
    # TEST 1: Custom evaluator with mateIn2 puzzles
    test_puzzles(
        puzzles=mate2_puzzles,
        test_name="TEST 1: CUSTOM EVALUATOR - MATE-IN-2 PUZZLES",
        use_engine_eval=False,
        max_depth=10
    )
    
    # TEST 2: Stockfish evaluator with mateIn2 puzzles (same puzzles)
    test_puzzles(
        puzzles=mate2_puzzles,
        test_name="TEST 2: STOCKFISH EVALUATOR - MATE-IN-2 PUZZLES",
        use_engine_eval=True,
        max_depth=10
    )
    
    # ========================================================================
    # TEST 3 & 4: Random mate puzzles (same puzzles for both tests)
    # ========================================================================
    print("\n" + "="*70)
    print("PREPARING RANDOM MATE PUZZLES")
    print("="*70)
    
    print("\nSelecting random mate puzzles from dataset...")
    print("(Filtering for puzzles with 'mate' theme, sampling without converting entire dataset)")
    
    # Efficiently sample random MATE puzzles without converting entire dataset
    random_puzzles = []
    max_to_check = 5000  # Check up to 5000 puzzles to find 5 mate puzzles
    
    # Iterate and collect mate puzzles
    collected = []
    checked = 0
    for puzzle in ds['train']:
        checked += 1
        if checked > max_to_check:
            break
        
        # Check if this puzzle has 'mate' in themes
        themes = puzzle.get('Themes', []) if isinstance(puzzle, dict) else []
        if 'mate' in themes:
            collected.append(puzzle)
            if len(collected) >= 5 * 2:  # Collect extra for random sampling
                break
        
        if checked % 500 == 0 and checked > 0:
            print(f"  Checked {checked:,} puzzles, found {len(collected)} mate puzzles...")
    
    if len(collected) >= 5:
        random_puzzles = random.sample(collected, 5)
    else:
        random_puzzles = collected
        print(f"  Warning: Only found {len(collected)} mate puzzles, using all of them")
    
    print(f"Selected {len(random_puzzles)} random mate puzzles for tests 3 & 4\n")
    
    # TEST 3: Custom evaluator with random mate puzzles
    test_puzzles(
        puzzles=random_puzzles,
        test_name="TEST 3: CUSTOM EVALUATOR - RANDOM MATE PUZZLES",
        use_engine_eval=False,
        max_depth=10
    )
    
    # TEST 4: Stockfish evaluator with random mate puzzles (same puzzles)
    test_puzzles(
        puzzles=random_puzzles,
        test_name="TEST 4: STOCKFISH EVALUATOR - RANDOM MATE PUZZLES",
        use_engine_eval=True,
        max_depth=10
    )
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)

