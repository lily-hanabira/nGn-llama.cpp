@echo off
title nGn MoE PoC - quick bench
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bench-phase-a.ps1" -Quick
pause
