# syntax=docker/dockerfile:1
FROM continuumio/miniconda3:latest
WORKDIR /app
COPY environment.yml .
RUN conda config --set channel_priority strict
RUN conda env create --name test_docker_final --file=environment.yml
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "test_docker_final", "/bin/bash", "-c"]
