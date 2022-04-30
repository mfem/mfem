FROM ghcr.io/rse-ops/cuda-ubuntu-20.04:cuda-11.0.3

# docker build -t ghcr.io/mfem/mfem-ubuntu-base .

RUN apt-get update && \
    apt-get install -y unzip gfortran && \
    spack compiler find && \
    apt-get install -y libcurl4-openssl-dev libssl-dev

# /code is the working directory for code
WORKDIR /code
COPY . /code

# This is for a spack environment/view to install from there
WORKDIR /opt/mfem-env
RUN . /opt/spack/share/spack/setup-env.sh && \
    spack env create -d . && \
    echo "  concretization: together" >> spack.yaml && \
    spack env activate . && \
    spack develop --path /code mfem@master+examples+miniapps && \
    spack add mfem@master+examples+miniapps && \
    spack install

# ensure mfem always on various paths
RUN cd /opt/mfem-env && \
    spack env activate --sh -d . >> /etc/profile.d/z10_spack_environment.sh

# The user will see the view on shell into the container
WORKDIR /opt/mfem-env/.spack-env/view/
ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l", "-c"]
