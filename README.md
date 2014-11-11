straighten
==========

Tool, um Videos von gekrümmten Wänden zu entzerren

Mit dem Makefile wird das Programm kompiliert, wobei die Optimierungsflags ggf. auf den aktuellen Prozessor angepasst werden müssen.

Das Makefile.proceed kann direkt alle .MTS-Dateien im aktuellen Ordner verarbeiten.

Der Aufruf ist: make [-j [n]] [POS=a] [FILE=000x.MTS] [MAIN=./main]

Sämtliche Parameter sind optional, es bietet sich an, direkt im Makefile den Pfad MAIN zu setzen.

mit [-j [n]] kann die Anzahl der gleichzeitig zu laufenden Jobs festgelegt werden, ohne n werden beliebig viele Jobs gemacht.

Mit FILE=000x.MTS POS=a wird die Position gesetzt, an der das die Maske genommen wird. Standardmäßig wird das erste Frame der ersten Datei genommen, falls nicht schon eine mask.png existiert.

Die Ausgabe erfolgt schließlich in der Datei ''all.mp4''
