MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ENV_DIR := $(MAKEFILE_DIR)env

setup:
	@if [ -d "$(ENV_DIR)" ]; then \
		echo "Virtual environment already exists."; \
		$(MAKE) update; \
		exit 0;\
	fi
	@echo "Creating virtual environment"
	python -m venv $(ENV_DIR)
	. $(ENV_DIR)/bin/activate && pip install -r $(MAKEFILE_DIR)requirements.txt

update:
	@if [ ! -d "$(ENV_DIR)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Upgradinocrg dependencies from requirements.txt"
	. $(ENV_DIR)/bin/activate && pip install --upgrade -r $(MAKEFILE_DIR)requirements.txt

clean:
	rm -rf $(ENV_DIR)
