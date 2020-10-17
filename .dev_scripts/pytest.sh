coverage run --branch --source mmtrack -m pytest tests/
coverage xml
coverage report -m
