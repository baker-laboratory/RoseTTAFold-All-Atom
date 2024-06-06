FROM mambaorg/micromamba:1.5.0 as micromamba
FROM nvidia/cuda:11.8.0-base-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

# Create root owned env: https://github.com/mamba-org/micromamba-docker/blob/main/examples/add_micromamba/Dockerfile
USER root
ENV MAMBA_USER=root
ENV MAMBA_USER_ID=0
ENV MAMBA_USER_GID=0
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    aria2 \
    build-essential \
    curl \
    git \
    tar \
    wget \
    unzip \
	vim \
    && rm -rf /var/lib/apt/lists/* \
    && micromamba install --name base -y python=3.11 -c conda-forge \
    && micromamba clean --all --yes

# DPF: everything below this line will have micromamba env activated
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /opt
ADD . /opt/RoseTTAFold-All-Atom/

RUN pip --no-cache-dir install -e ./RoseTTAFold-All-Atom/rf2aa/SE3Transformer --no-deps

WORKDIR /opt/RoseTTAFold-All-Atom

ADD install_dependencies.sh /opt/RoseTTAFold-All-Atom/install_dependencies.sh
RUN bash /opt/RoseTTAFold-All-Atom/install_dependencies.sh

RUN wget https://ftp.ncbi.nlm.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26/blast-2.2.26-x64-linux.tar.gz
RUN mkdir -p blast-2.2.26
RUN tar -xf blast-2.2.26-x64-linux.tar.gz -C blast-2.2.26
RUN cp -r blast-2.2.26/blast-2.2.26/ blast-2.2.26_bk
RUN rm -r blast-2.2.26
RUN mv blast-2.2.26_bk/ blast-2.2.26

#Get The Weights. We Opted to store the weights in the database at /databases/weights/RFAA_paper_weights.pt To reduce the docker image size. Leaving this here for legacy reasons. 
#RUN wget http://files.ipd.uw.edu/pub/RF-All-Atom/weights/RFAA_paper_weights.pt

ADD environment.yaml /opt/RoseTTAFold-All-Atom/environment.yaml

RUN CONDA_OVERRIDE_CUDA="11.8" micromamba install -y -n base -f /opt/RoseTTAFold-All-Atom/environment.yaml && \
    micromamba clean --all --yes

ENV DB_UR30=/mnt/databases/rfaa/latest/UniRef30_2020_06/UniRef30_2020_06
ENV DB_BFD=/mnt/databases/rfaa/latest/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt
ENV BLASTMAT=/opt/RoseTTAFold-All-Atom/blast-2.2.26/data/
ENTRYPOINT ["micromamba", "run", "-n", "base"]
