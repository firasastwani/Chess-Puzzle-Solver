# ChessAI - Minimax Puzzle Solver

## Requirements

- Stockfish chess engine (must be installed separately)
- Virtual environment 

## Setup Instructions

### 1. Install Stockfish

**macOS (using Homebrew):**

```bash
brew install stockfish
```

**Windows:**
Download from [Stockfish website](https://stockfishchess.org/download/) and add to PATH, or install via package manager.


### 2. Create Virtual Environment

```bash
python3 -m venv venv
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Stockfish Path

The code will auto-detect Stockfish at common locations:

- `/opt/homebrew/bin/stockfish` (macOS with Homebrew)
- System PATH

If Stockfish is installed elsewhere, you can specify the path when calling functions.

## Usage

```
### Testing on Lichess Dataset

#### Test Endgame Puzzles

Run the test script to solve random endgame mate-in-2 puzzles:

```bash
python test_puzzles.py
```

#### Test High-Rated Puzzles

Test on higher-rated puzzles:

```bash
python test_high_rated_puzzles.py
```
