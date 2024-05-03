FROM mambaorg/micromamba:1.5.0 as micromamba
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
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
# DPF torch before dgl
# Must use -e install as we want to use schedule files from ISOG3
#This is copied from the RoseTTAFold-All-Atom Image which was copied from the rf-diffusion image. I haven't tried to check which are absolutely necessary
RUN pip --no-cache-dir install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip --no-cache-dir install dgl -f https://data.dgl.ai/wheels/cu118/repo.html \
    && pip --no-cache-dir install dglgo -f https://data.dgl.ai/wheels-test/repo.html \
    && pip --no-cache-dir install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && pip --no-cache-dir install -e ./RoseTTAFold-All-Atom/rf2aa/SE3Transformer --no-deps  \
    && pip --no-cache-dir install hydra-core pyrsistent jedi omegaconf icecream scipy opt_einsum opt_einsum_fx e3nn wandb decorator pynvml \
    && mkdir -p $CONDA_PREFIX/etc/conda/activate.d \
    && echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh \
    && echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh \
    && for x in cublas cuda_cupti cuda_runtime cufft cusolver cusparse; do CURVAR=$(dirname $(python -c "import nvidia.$x;print(nvidia.$x.__file__)")) && echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$CURVAR/lib" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh; done

WORKDIR /opt/RoseTTAFold-All-Atom

ADD install_dependencies.sh /opt/RoseTTAFold-All-Atom/install_dependencies.sh
RUN bash /opt/RoseTTAFold-All-Atom/install_dependencies.sh

#Get The Weights
RUN wget http://files.ipd.uw.edu/pub/RF-All-Atom/weights/RFAA_paper_weights.pt

ADD environment.yaml /opt/RoseTTAFold-All-Atom/environment.yaml

RUN CONDA_OVERRIDE_CUDA="11.8" micromamba install -y -n base -f /opt/RoseTTAFold-All-Atom/environment.yaml && \
    micromamba clean --all --yes

#Move this to the top later
ENV DB_DIR=/mnt/databases/
ENV DB_UR30=/mnt/databases/UniRef30_2020_06/UniRef30_2020_06
ENV DB_BFD=/mnt/databases/bfd/
ENTRYPOINT ["micromamba", "run", "-n", "base"]
