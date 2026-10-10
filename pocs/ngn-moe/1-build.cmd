@echo off
title nGn MoE PoC - build
call "%~dp0build-poc.cmd" %*
echo exit code %ERRORLEVEL%
pause
