CC = gcc
CFLAGS = -O2 -Wall

all: min_cost_flow kmp

min_cost_flow: src/min_cost_flow.c
	$(CC) $(CFLAGS) src/min_cost_flow.c -o min_cost_flow

kmp: src/kmp.c
	$(CC) $(CFLAGS) src/kmp.c -o kmp

clean:
	rm -f min_cost_flow kmp

test_mcf: min_cost_flow
	./min_cost_flow data/mcf_input.txt

test_kmp: kmp
	./kmp --benchmark

report: REPORT.md
	python3 scripts/plot_kmp.py
	pandoc REPORT.md -o REPORT.pdf --pdf-engine=xelatex \
		-V mainfont="Noto Serif CJK SC" \
		-V monofont="Noto Sans Mono CJK SC" \
		-V fontsize=12pt \
		-V linestretch=1.5 \
		-V papersize=a4 \
		--number-sections
