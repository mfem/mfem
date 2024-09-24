FROM ghcr.io/mfem/mfem-ubuntu-base:latest as builder

# docker build -t ghcr.io/mfem/mfem-ubuntu .

COPY ./config/docker/spack.yaml /opt/mfem-env/spack.yaml
RUN apt-get install -y python3 && \
    cd /opt/mfem-env && \
    . /opt/spack/share/spack/setup-env.sh && \
    spack env activate . && \
    spack env view regenerate

FROM ubuntu:22.04

COPY --from=builder /opt/view /opt/view
COPY --from=builder /opt/mfem-view /opt/mfem-view

RUN apt-get update && \
    apt-get install -y unzip gfortran && \
    apt-get install -y libcurl4-openssl-dev libssl-dev

ENV PATH=$PATH:/opt/mfem-view/bin
ENV DEBIAN_FRONTEND=noninteractive

# The user will see the view on shell into the container
WORKDIR /opt/mfem-view
