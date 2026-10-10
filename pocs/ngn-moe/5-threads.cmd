@echo off
title nGn MoE PoC - threads
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bench-threads.ps1"
pause
