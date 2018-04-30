@setlocal
@echo off

start "toki1 intensity"  /HIGH  %~dp0\toki1.cmd -i %1 intensity  ^&^& exit
start "toki1 interarrival" /HIGH  %~dp0\toki1.cmd -i %1 interarrival  ^&^& exit
start "toki1 stats" /HIGH  %~dp0\toki1.cmd -i %1 stats ^> stats.txt  ^&^& exit
start "toki1 packetcountcorrelation" /HIGH  %~dp0\toki1.cmd -i %1 packetcountcorrelation  ^&^& exit
start "toki1 arrtimecorrelation" /HIGH  %~dp0\toki1.cmd -i %1 arrtimecorrelation  ^&^& exit
:: idc and idi calculations happen in the same run because of memoization of windowed sums
start "toki1 idc" /W /HIGH  %~dp0\toki1.cmd -i %1  idi idc  ^&^& exit

endlocal
