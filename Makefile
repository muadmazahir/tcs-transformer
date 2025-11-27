lint-fix:
	poetry run ruff check --fix tcs_transformer/

lint:
	poetry run ruff check tcs_transformer/

format:
	poetry run ruff format tcs_transformer/
