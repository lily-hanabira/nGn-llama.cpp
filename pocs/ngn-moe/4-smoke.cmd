@echo off
title nGn MoE PoC - 32K/8K smoke
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0smoke-server.ps1" %*
pause
