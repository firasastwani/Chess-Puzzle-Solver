# Minimax Algorithm Issues Analysis

## Test Results Summary

- **Success Rate**: 40% (2/5 puzzles solved)
- **Pattern**: First move is usually correct, but second move fails

## Critical Issues Found

### 1. **TOO SHALLOW EVALUATION DEPTH** ⚠️ CRITICAL

**Location**: `main.py:377`

```python
return evaluate_position_engine(board, engine, depth_limit=1, time_limit=0.01), None
```

**Problem**:

- Stockfish is only given **1 ply of search** at depth 0
- This is essentially a static evaluation, not a tactical search
- For mate-in-2 puzzles, Stockfish needs to see **4-5 plies** to find forced mates
- Even though Stockfish is "perfect", giving it only 1 ply means it can't see tactical sequences

**Impact**: The algorithm can't see the full mate sequence because the evaluation is too shallow.

### 2. **MAXIMIZING PARAMETER LOGIC ERROR** ⚠️ CRITICAL

**Location**: `main.py:505`

```python
score, best_move = minimax(board, depth, float('-inf'), float('inf'), board.turn, ...)
```

**Problem**:

- `board.turn` is `chess.WHITE` (True) or `chess.BLACK` (False)
- The minimax function expects `maximizing` to indicate whether we're maximizing the score
- But the score is always from **White's perspective** (positive = good for white)
- When it's Black's turn, we should be "maximizing for Black" which means **minimizing from White's perspective**
- Currently, `board.turn == chess.WHITE` (True) means maximizing, but `board.turn == chess.BLACK` (False) also gets treated as minimizing, which is correct by accident, BUT...

**The real issue**: The evaluation function at line 172 in `custom_eval.py` already flips the score:

```python
return int(score) if board.turn == chess.WHITE else -int(score)
```

This creates a **double-flipping problem** or perspective confusion.

### 3. **INSUFFICIENT SEARCH DEPTH**

**Location**: `main.py:505` - depth=6

**Problem**:

- For mate-in-2, we need to see: Our move → Opponent → Our move → Opponent → Checkmate
- That's 5 plies minimum
- With depth=6, we should see it, BUT the shallow depth_limit=1 at leaves means we're not really searching 6 plies deep
- The algorithm reaches depth 0 quickly and then only does 1-ply evaluations

### 4. **SCORE PERSPECTIVE INCONSISTENCY**

**Location**: `custom_eval.py:172` and `main.py:391,407`

**Problem**:

- `evaluate_position_engine` returns scores from the current player's perspective (flips based on `board.turn`)
- But minimax expects scores from a consistent perspective (typically White's)
- This causes confusion when comparing scores across different turns

## Why It Fails on Second Move

Looking at the failures:

1. **Puzzle 1**: Found `h2h8` correctly, but then `h8b8` instead of `h1h7`
2. **Puzzle 2**: Found `b3g3` correctly, but then `h6g7` instead of `g3g7`
3. **Puzzle 3**: Found `d5e7` correctly, but then `e7g6` instead of `f2f8`

**Root Cause**: After the first move, the algorithm is searching from a new position. The shallow depth_limit=1 means it can't see that the second move leads to mate. It evaluates positions statically and picks moves that look good in the short term, but don't complete the mate sequence.

## Solutions

### Priority 1: Fix Shallow Evaluation (MOST CRITICAL)

**Change in `main.py:377`**:

```python
# CURRENT (WRONG):
return evaluate_position_engine(board, engine, depth_limit=1, time_limit=0.01), None

# SHOULD BE:
return evaluate_position_engine(board, engine, depth_limit=4, time_limit=0.1), None
```

**Why**: Stockfish needs to see 4-5 plies ahead to find forced mates. With depth_limit=1, it's essentially blind to tactical sequences.

### Priority 2: Fix Score Perspective

The evaluation function flips scores based on `board.turn`, but minimax expects consistent perspective. Two options:

**Option A**: Keep evaluation from current player's perspective, but ensure minimax handles it correctly
**Option B**: Always evaluate from White's perspective, and let minimax handle the flipping

### Priority 3: Increase Main Search Depth

For mate-in-2 puzzles, use depth=8 or depth=10 to ensure we see the full sequence.

### Priority 4: Use Mate Search Mode

Stockfish can search specifically for mates, which is faster and more accurate for puzzle solving.
