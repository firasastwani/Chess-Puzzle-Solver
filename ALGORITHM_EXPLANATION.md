# Algorithm Explanation: Forward Search Puzzle Solver

This document explains how each component of the algorithm in `main.py` works together to solve chess puzzles.

## Overview

The algorithm uses **forward search** with **known opponent responses** to find checkmate sequences. Unlike adversarial algorithms (like minimax), it doesn't assume the opponent plays optimally - it uses the actual moves from the puzzle dataset.

## Component Breakdown

### 1. `load_puzzle_board(fen_string)` (Lines 16-25)

**Purpose**: Initialize the chess board from the puzzle's starting position.

**How it works**:
- Takes a FEN (Forsyth-Edwards Notation) string
- Creates a `chess.Board` object representing the initial position
- This is the starting point for the search

**Role in solving**:
- Sets up the initial game state before the opponent's first move
- Example: `"r1bqk2r/pp1nbNp1/2p1p2p/8/2BP4/1PN3P1/P3QP1P/3R1RK1 b kq - 0 19"`

---

### 2. `parse_puzzle_moves(puzzle_data)` (Lines 256-293)

**Purpose**: Parse the puzzle's move sequence and separate our moves from opponent responses.

**How it works**:
- Puzzle format: `"e8f7 e2e6 f7f8 e6f7"`
  - Index 0 (`e8f7`): Opponent's first move (the blunder)
  - Index 1 (`e2e6`): Our first move (to find)
  - Index 2 (`f7f8`): Opponent's response (from dataset)
  - Index 3 (`e6f7`): Our second move (to find)

**Returns**:
- `board_after_opponent_first_move`: Board state after opponent's blunder
- `our_expected_moves`: List of our moves to find `["e2e6", "e6f7"]`
- `opponent_responses`: List of opponent responses `["f7f8"]`

**Role in solving**:
- **Critical**: Separates what we need to find (our moves) from what we know (opponent responses)
- Applies the opponent's first move to get the starting position
- Provides the known opponent responses that the search will use

---

### 3. `order_moves(board, moves, previous_move=None)` (Lines 159-249)

**Purpose**: Prioritize moves to try the most promising ones first, dramatically reducing search time.

**How it works** - Assigns scores based on move characteristics:

1. **Checkmate detection** (score +1,000,000):
   - If a move gives checkmate, it's tried first
   - Highest priority - ends the search immediately

2. **Check detection** (score +100,000):
   - Moves that put the opponent in check
   - Very important for mate puzzles

3. **Tactical continuations** (score +5,000 to +50,000):
   - If we have a previous move, prioritize moves that continue the sequence
   - Example: `h2h8` → `h8h7` (same piece continuing)
   - Catches patterns like rook lifts and piece coordination

4. **Captures** (score +10,000 + piece value):
   - MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
   - Capturing a queen (900) is better than capturing a pawn (100)
   - Important for tactical sequences

5. **Center control** (score +10):
   - Small bonus for moving to central squares
   - Less important but helps with general play

**Returns**: Moves sorted by priority (highest score first)

**Role in solving**:
- **Massive efficiency gain**: Instead of trying 20+ moves randomly, tries checkmates first
- For mate-in-2 puzzles, often finds the solution in 2-3 nodes because:
  - First move is a check (high priority)
  - Second move is checkmate (highest priority)
- The tactical continuation bonus helps maintain sequences across multiple moves

**Example**:
```
Legal moves: [a1a2, b1c3, e2e6, f1f2, ...]
After ordering: [e2e6, f1f2, b1c3, a1a2, ...]
  → e2e6 is first because it gives check!
```

---

### 4. `forward_search(current_board, our_move_index, our_moves_so_far)` (Lines 350-432)

**Purpose**: The core recursive search algorithm that finds the checkmate sequence.

**How it works** - Recursive depth-first search:

#### Base Cases (Termination Conditions):

1. **Checkmate found** (Line 366):
   ```python
   if current_board.is_checkmate():
       return True, our_moves_so_far
   ```
   - If we've reached checkmate, we're done!
   - Returns the sequence of moves that led to checkmate

2. **All expected moves made** (Line 370):
   ```python
   if our_move_index >= expected_num_our_moves:
       return current_board.is_checkmate(), our_moves_so_far
   ```
   - If we've made all our moves, check if we're in checkmate
   - Prevents searching beyond the puzzle's expected length

3. **Max depth exceeded** (Line 374):
   ```python
   if our_move_index >= max_depth:
       return False, our_moves_so_far
   ```
   - Safety limit to prevent infinite search
   - Default: 10 moves

#### Recursive Search Process:

1. **Get ordered moves** (Line 378):
   ```python
   ordered_moves = order_moves(current_board, current_board.legal_moves, previous_move)
   ```
   - Gets all legal moves and orders them by priority
   - Tries best moves first

2. **Try each move** (Line 381):
   ```python
   for move in ordered_moves:
       current_board.push(move)  # Make our move
   ```

3. **Check for immediate checkmate** (Line 387):
   - If this move gives checkmate, return immediately
   - No need to search further

4. **Apply known opponent response** (Lines 391-407):
   ```python
   if our_move_index < len(opponent_responses):
       opponent_response = chess.Move.from_uci(opponent_responses[our_move_index])
       if opponent_response in current_board.legal_moves:
           current_board.push(opponent_response)
           found, solution = forward_search(current_board, our_move_index + 1, new_moves)
   ```
   - **Key difference from minimax**: Uses the known opponent response from the dataset
   - Not adversarial - doesn't assume opponent minimizes our score
   - Recursively searches for the next move

5. **Handle forced mate** (Lines 412-426):
   - If no more opponent responses expected, check if all opponent moves lead to checkmate
   - This handles mate-in-1 cases

6. **Backtrack** (Line 429):
   ```python
   current_board.pop()  # Undo our move
   ```
   - Undoes the move to try the next one
   - Essential for exploring all possibilities

**Role in solving**:
- **The heart of the algorithm**: Actually searches for the solution
- Uses known opponent responses (not adversarial search)
- Explores moves in priority order (from `order_moves`)
- Backtracks when a path doesn't lead to checkmate
- Returns as soon as checkmate is found

**Example Search Tree**:
```
Start position (after opponent's e8f7)
├── Try e2e6 (check - high priority)
│   └── Apply known response: f7f8
│       └── Try e6f7 (checkmate - highest priority!)
│           └── ✓ SOLUTION FOUND!
└── Try other moves (lower priority, won't be reached)
```

---

### 5. `solve_puzzle(puzzle_data, ...)` (Lines 296-468)

**Purpose**: Main entry point that orchestrates the entire solving process.

**How it works** - Step by step:

1. **Initialize** (Lines 313-334):
   - Reset node counter
   - Optionally initialize Stockfish engine (if using engine evaluation)

2. **Parse puzzle** (Line 337):
   ```python
   board, our_expected_moves, opponent_responses = parse_puzzle_moves(puzzle_data)
   ```
   - Gets the starting position and separates our moves from opponent responses

3. **Start search** (Line 436):
   ```python
   found, solution_moves = forward_search(board, 0, [])
   ```
   - Calls `forward_search` starting from the initial position
   - `our_move_index=0`: Looking for our first move
   - `our_moves_so_far=[]`: No moves played yet

4. **Report results** (Lines 439-459):
   - Prints solution if found
   - Shows nodes explored, time taken
   - Verifies solution matches expected

**Role in solving**:
- **Orchestrator**: Ties all components together
- Sets up the problem (parses puzzle)
- Runs the search (calls `forward_search`)
- Reports the results

---

## How It All Works Together

### Complete Flow for a Mate-in-2 Puzzle:

1. **Input**: Puzzle with FEN and moves `"e8f7 e2e6 f7f8 e6f7"`

2. **`load_puzzle_board()`**: Creates initial board from FEN

3. **`parse_puzzle_moves()`**:
   - Applies opponent's first move: `e8f7`
   - Separates: our moves = `["e2e6", "e6f7"]`, opponent = `["f7f8"]`

4. **`solve_puzzle()`** calls **`forward_search()`**:
   - Position: After `e8f7`
   - Looking for: Our first move

5. **`forward_search()`**:
   - Gets legal moves: `[e2e6, f1f2, ...]`
   - **`order_moves()`** prioritizes: `e2e6` first (gives check!)
   - Tries `e2e6`:
     - Makes move: `board.push(e2e6)`
     - Checks: Not checkmate yet
     - Applies known opponent response: `f7f8`
     - Recursively calls `forward_search()` for next move
     - In recursive call:
       - Gets legal moves: `[e6f7, ...]`
       - **`order_moves()`** prioritizes: `e6f7` first (checkmate!)
       - Tries `e6f7`:
         - Makes move: `board.push(e6f7)`
         - Checks: **CHECKMATE!** ✓
         - Returns `True, [e2e6, e6f7]`

6. **Solution found**: `["e2e6", "e6f7"]` leads to checkmate!

---

## Key Design Decisions

### Why Forward Search (Not Minimax)?

1. **Known opponent responses**: Puzzles have predetermined opponent moves
2. **No adversarial assumption**: Don't need to assume opponent minimizes our score
3. **Efficiency**: Only explores the actual sequence, not all opponent possibilities
4. **Accuracy**: Uses actual puzzle responses, not theoretical optimal play

### Why Move Ordering?

1. **Massive speedup**: Tries checkmates first (often finds solution in 2-3 nodes)
2. **Tactical awareness**: Prioritizes checks, captures, continuations
3. **Pattern recognition**: Tactical continuation bonus catches common sequences

### Why Recursive Search?

1. **Systematic exploration**: Tries all moves systematically
2. **Backtracking**: Undoes moves that don't lead to checkmate
3. **Depth-first**: Finds solutions quickly (doesn't need to explore entire tree)

---

## Performance Characteristics

- **Time complexity**: O(b^d) where b = branching factor, d = depth
  - But move ordering reduces effective b dramatically
  - For mate-in-2: Often finds solution in 2-4 nodes

- **Space complexity**: O(d) for recursion stack
  - Very memory efficient

- **Typical performance**:
  - Mate-in-2 puzzles: 2-4 nodes, <0.01s
  - Mate-in-3 puzzles: 10-50 nodes, <0.1s
  - Complex puzzles: 100-1000+ nodes, <1s

---

## Summary

The algorithm works by:
1. **Parsing** the puzzle to separate our moves from opponent responses
2. **Ordering** moves to try the most promising first
3. **Searching** recursively, using known opponent responses
4. **Backtracking** when a path doesn't lead to checkmate
5. **Returning** as soon as checkmate is found

The combination of **forward search** (using known responses) and **move ordering** (trying best moves first) makes it extremely efficient for puzzle solving, often finding solutions in just 2-3 nodes explored.

