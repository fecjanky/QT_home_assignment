@setlocal
@echo off

start "toki1 intensity" /B /AFFINITY 1 /HIGH  %~dp0\toki1.cmd -i %1 intensity
start "toki1 interarrival" /B /AFFINITY 2 /HIGH  %~dp0\toki1.cmd -i %1 interarrival
start "toki1 stats" /B /AFFINITY 4 /HIGH  %~dp0\toki1.cmd -i %1 stats ^> stats.txt
start "toki1 packetcountcorrelation" /B  /AFFINITY 8 /HIGH  %~dp0\toki1.cmd -i %1 packetcountcorrelation
start "toki1 arrtimecorrelation" /B /AFFINITY 1 /HIGH  %~dp0\toki1.cmd -i %1 arrtimecorrelation
start "toki1 idi" /B /AFFINITY 2 /HIGH  %~dp0\toki1.cmd -i %1 idi
start "toki1 idc" /B /W /AFFINITY 4 /HIGH  %~dp0\toki1.cmd -i %1 idc 

endlocal
