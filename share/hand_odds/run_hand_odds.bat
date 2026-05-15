@echo off
cd /d "%~dp0"
python balatro_hand_odds.py
if errorlevel 1 pause

