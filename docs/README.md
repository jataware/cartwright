# Cartwright Github Pages

## Developing

We recommend running Jekyll in Docker to build the docs locally for editing and development purposes.

First, uncomment line 13 in `_config.yaml`. Then run:

```
docker run -it --rm -v "$PWD":/usr/src/app -p "4000:4000" jataware/darpa-askem-docs
```

> Note: for the Github Pages to build, line 13 in `_config.yaml` must be commented out. Before committing to this repository, please ensure that is commented out. It should only be _uncommented_ for development purposes.