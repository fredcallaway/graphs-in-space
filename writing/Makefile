panzer = panzer -f markdown

all: thesis

thesis: header.md introduction.md graphical_models.md vsa.md vectorgraph.md numila.md conclusion.md
	cat $+ | $(panzer) -o $@.tex

introduction: header.md introduction.md
	cat $+ | $(panzer) -o $@.tex

graphical_models: header.md graphical_models.md
	cat $+ | $(panzer) -o $@.tex

vsa: header.md vsa.md
	cat $+ | $(panzer) -o $@.tex

vectorgraph: header.md vectorgraph.md
	cat $+ | $(panzer) -o $@.tex

numila: header.md numila.md
	cat $+ | $(panzer) -o $@.tex

conclusion: header.md conclusion.md
	cat $+ | $(panzer) -o $@.tex

operations: commands.tex operations.md
	cat $+ | pandoc -f markdown -o $@.pdf

slides: slides.md
	 $(panzer) -t beamer slides.md -o slides.tex
