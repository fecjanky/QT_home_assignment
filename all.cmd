@setlocal
@echo off

start "toki1 intensity" /B  /HIGH  %~dp0\toki1.cmd -i %1 intensity
start "toki1 interarrival" /B /HIGH  %~dp0\toki1.cmd -i %1 interarrival
start "toki1 stats" /B /HIGH  %~dp0\toki1.cmd -i %1 stats ^> stats.txt
start "toki1 packetcountcorrelation" /B  /HIGH  %~dp0\toki1.cmd -i %1 packetcountcorrelation
start "toki1 arrtimecorrelation" /B /HIGH  %~dp0\toki1.cmd -i %1 arrtimecorrelation
start "toki1 idi" /B /HIGH  %~dp0\toki1.cmd -i %1 idi
start "toki1 idc" /B /W /HIGH  %~dp0\toki1.cmd -i %1 idc 

endlocal
