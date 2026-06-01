.PHONY: all
.PHONY: clean

all: prueba-3-parte-escrita.pdf

%.pdf: %.tex setup.tex header.tex questions.tex
	latexmk -shell-escape -pdflua $<

clean:
	rm -f *.aux *.log *.out *.snm *.toc *.vrb *.nav *.synctex.gz *.blg *.bbl *.fdb_latexmk *.fls *.ind *.idx *.ilg *.bcf *.run.xml

