#!/bin/bash

# Build the Docker image
docker build -t llm-eval-test .

# Run the unit tests within the container to verify the base setup.
# This ensures the Dockerfile and requirements are correct.
docker run llm-eval-test

echo "Setup complete.  Image 'llm-eval-test' built and tested."
echo "To run evaluations, use:"
echo "  docker run -v /path/to/your/target_model:/app/target_model llm-eval-test python main.py"