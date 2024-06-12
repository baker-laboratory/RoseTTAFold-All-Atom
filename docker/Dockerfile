FROM ghcr.io/washu-it-ris/novnc:ubuntu20.04_cuda11.6_xfce

RUN apt-get -q update \ 
  && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y && \
  rm -rf /var/lib/apt/lists/* 

WORKDIR /app/RoseTTAFold/RoseTTAFold-All-Atom

COPY examples/ examples/.
COPY img/ img/.
COPY input_prep/ input_prep/.
COPY rf2aa/ rf2aa/.
COPY environment.yaml .
COPY install_dependencies.sh .
COPY make_msa.sh .
COPY __init__.py .
COPY LICENSE .

RUN chmod a+rwx /app/RoseTTAFold/RoseTTAFold-All-Atom && \
    chmod +rx /app/RoseTTAFold/RoseTTAFold-All-Atom/input_prep/make_ss.sh make_msa.sh

COPY docker/build_dependencies/signalp-6.0h.fast.tar.gz .

COPY docker/build_dependencies/RFAA_paper_weights.pt .

RUN rm -rf /app/RoseTTAFold/RoseTTAFold-All-Atom/docker/

WORKDIR /app/RoseTTAFold/

RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" && \
    chsh -s /bin/bash && \
    bash Mambaforge-$(uname)-$(uname -m).sh -p /app/RoseTTAFold/mambaforge -b  && \
    rm /app/RoseTTAFold/Mambaforge-$(uname)-$(uname -m).sh && \
    . /root/.bashrc 

WORKDIR /app/RoseTTAFold/RoseTTAFold-All-Atom

RUN export PATH="${PATH}:/app/RoseTTAFold/mambaforge/condabin:/app/RoseTTAFold/mambaforge/bin" && \
    conda update -n base -c conda-forge conda && \
    CONDA_OVERRIDE_CUDA="11.6" mamba env create -f environment.yaml && \
    conda init && \
    . /root/.bashrc && \
    conda activate RFAA && \
    conda clean -afy && \
    pip3 install --no-cache-dir -r rf2aa/SE3Transformer/requirements.txt && \
    python3 rf2aa/SE3Transformer/setup.py install && \
    signalp6-register signalp-6.0h.fast.tar.gz && \
    rm signalp-6.0h.fast.tar.gz && \
    mv $CONDA_PREFIX/lib/python3.10/site-packages/signalp/model_weights/distilled_model_signalp6.pt $CONDA_PREFIX/lib/python3.10/site-packages/signalp/model_weights/ensemble_model_signalp6.pt && \
    bash install_dependencies.sh

RUN wget https://ftp.ncbi.nlm.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26/blast-2.2.26-x64-linux.tar.gz && \
    mkdir -p blast-2.2.26 && \
    tar -xf blast-2.2.26-x64-linux.tar.gz -C blast-2.2.26 && \
    cp -r blast-2.2.26/blast-2.2.26/ blast-2.2.26_bk && \
    rm -r blast-2.2.26 && \
    mv blast-2.2.26_bk/ blast-2.2.26 && \
    rm blast-2.2.26-x64-linux.tar.gz csblast-2.2.3.tar.gz

ENTRYPOINT ["/bin/bash"]
