SPHINXOPTS    ?=
SPHINXBUILD   ?= python3 -m sphinx
SOURCEDIR     = source
BUILDDIR      = _build

.DEFAULT_GOAL := help

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean precommit check serve

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html: Makefile
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

serve: html
	@echo "Open http://localhost:8000/ in your browser"
	@cd $(BUILDDIR)/html && python3 -m http.server $(PORT)

clean:
	rm -rf $(BUILDDIR) $(EXAMPLES_DIR)/$(BUILD_DIR)

EXAMPLES_DIR = examples
BUILD_DIR    = build

examples:
	@mkdir -p $(EXAMPLES_DIR)/$(BUILD_DIR)
	cd $(EXAMPLES_DIR)/$(BUILD_DIR) && cmake .. && make
	@echo ""
	@echo "=== Examples built successfully ==="
	@ls -lh $(EXAMPLES_DIR)/$(BUILD_DIR)/vector_add $(EXAMPLES_DIR)/$(BUILD_DIR)/wmma_matmul
