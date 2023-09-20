# Build image
DOCKER_BUILDKIT=1 docker build --target=builder -t drifts:deckard .   

# Run the container
docker container run -p 127.0.0.1:8888:8888 --mount type=bind,source="$(pwd)",target=/dev -it --entrypoint /bin/bash drifts:deckard 

# Inside the container
poetry shell
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
