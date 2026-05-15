Botlatro Hand Odds
==================

This is a standalone Tkinter GUI for estimating Balatro hand-type odds.

How to run
----------

On Windows, double-click:

    run_hand_odds.bat

Or run from a terminal:

    python balatro_hand_odds.py

Requirements
------------

- Python 3.10 or newer.
- Tkinter, which is included with the normal python.org Windows installer.
- No third-party packages are needed.

What it estimates
-----------------

The app runs Monte Carlo simulations for each hand type and shows:

- Opening: probability the starting hand already contains that playable hand type.
- After Discards: probability after spending the configured discards.
- After Hands: probability after also using spare played hands as redraws.

The default is 2,000 trials. Increase the
Trials field for smoother estimates.

Hand types use Balatro's exact classification for the cards you would play.
For example, five copies of the same rank all in one suit count as Flush Five,
not Five of a Kind. Five of a Kind needs five same-rank cards that are not all
the same suit.

The redraw logic is a target-specific hunting heuristic. For example, if it is
hunting a straight and the hand already has four cards in a straight window, it
keeps those four cards and only discards the non-straight cards.
