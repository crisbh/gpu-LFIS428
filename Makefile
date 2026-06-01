# Define the list of slides (all .md files in the slides directory)
SLIDES := $(wildcard slides/*.md)

# Define the output directory for HTML slides
OUTPUT_DIR := static/slides

# Intermediate dir holding slides after @include expansion (fed to marp)
BUILD_DIR := .slides-build

# Convert .md files to .html
HTML_SLIDES := $(patsubst slides/%.md, $(OUTPUT_DIR)/%.html, $(SLIDES))

# Expand @include directives (LaTeX-style code inclusion) before marp
EXPAND := python3 scripts/expand-includes.py

# Define Marp command
MARP_CMD := marp --html --theme slides/themes/curso.css
HUGO_CMD := hugo server --minify --theme hugo-book

# Build website with hugo
site: slides
	$(HUGO_CMD)

# Default target: Build all slides
all: slides site

# Rule to convert .md to .html (Build Slides)
#
slides: $(HTML_SLIDES)

# Expand @include directives into a temp .md, then render that with marp.
# (marp passes relative image/link URLs through verbatim, so the temp file's
# location does not affect the generated HTML.)
$(OUTPUT_DIR)/%.html: slides/%.md scripts/expand-includes.py
	@mkdir -p $(OUTPUT_DIR) $(BUILD_DIR)
	@$(EXPAND) $< > $(BUILD_DIR)/$*.md
	$(MARP_CMD) $(BUILD_DIR)/$*.md --output $@

# Clean all generated files
clean:
	rm -rf $(OUTPUT_DIR)/*.html
	rm -rf $(BUILD_DIR)
	rm -rf public/

.PHONY: all clean site
