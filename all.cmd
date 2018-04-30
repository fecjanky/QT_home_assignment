@setlocal
@echo off

start "toki1 intensity"  /HIGH  %~dp0\toki1.cmd -i %* intensity  ^&^& exit
start "toki1 interarrival" /HIGH  %~dp0\toki1.cmd -i %* interarrival  ^&^& exit
start "toki1 stats" /HIGH  %~dp0\toki1.cmd -i %* stats ^> stats.txt  ^&^& exit
start "toki1 packetcountcorrelation" /HIGH  %~dp0\toki1.cmd -i %* packetcountcorrelation  ^&^& exit
start "toki1 arrtimecorrelation" /HIGH  %~dp0\toki1.cmd -i %* arrtimecorrelation  ^&^& exit
:: idc and idi calculations happen in the same run because of memoization of windowed sums
start "toki1 idc" /W /HIGH  %~dp0\toki1.cmd -i %*  idi idc  ^&^& exit

endlocal
