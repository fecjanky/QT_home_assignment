@setlocal
@echo off

start "toki1 intensity" /AFFINITY 1 /HIGH  %~dp0\toki1.cmd -i %1 intensity
start "toki1 interarrival" /AFFINITY 2 /HIGH  %~dp0\toki1.cmd -i %1 interarrival
start "toki1 stats" /AFFINITY 4 /HIGH  %~dp0\toki1.cmd -i %1 stats ^> stats.txt
start "toki1 packetcountcorrelation" /B /W /AFFINITY /8 /HIGH  %~dp0\toki1.cmd -i %1 packetcountcorrelation

start "toki1 arrtimecorrelation" /AFFINITY 1 /HIGH  %~dp0\toki1.cmd -i %1 arrtimecorrelation
start "toki1 idi" /AFFINITY 2 /HIGH  %~dp0\toki1.cmd -i %1 idi
start "toki1 idc" /B /W /AFFINITY 4 /HIGH  %~dp0\toki1.cmd -i %1 idc 

::FOR %%A IN (intensity,interarrival,packetcountcorrelation) DO (
::	start "toki1 %%A" /B  %~dp0\toki1.cmd -i %1 %%A
::)
::
::FOR %%A IN (stats,arrtimecorrelation,idi,idc) DO (
::	start "toki1 %%A" /B  %~dp0\toki1.cmd -i %1 %%A
::)


endlocal
::start "toki1 intensity" /W /AFFINITY 1 %~dp0\toki1.cmd -i %1 intensity
::start "toki1 intensity" /W /AFFINITY 1 %~dp0\toki1.cmd -i %1 arrtimecorrelation
::start "toki1 arrtimecorrelation" %~dp0\toki1.cmd -i %1 arrtimecorrelation