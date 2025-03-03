FROM ubuntu:jammy

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  gpg \
  wget \
  git \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log}

# Install oneAPI DPC++
#
# oneAPI only has one stream end point
#
# https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2024-0/apt.html
RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
    gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
    tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    intel-oneapi-dpcpp-cpp-2024.0 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log}

WORKDIR /home/user

# Build oneDNN from source
#
# https://oneapi-src.github.io/oneDNN/v2/dev_guide_build.html
RUN git clone https://github.com/oneapi-src/oneDNN.git
WORKDIR /home/user/oneDNN/build

# Install dependencies needed to build oneDNN
#
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    build-essential \
    cmake \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log}

SHELL [ "/bin/bash", "-c" ]
ENV CC=icx
ENV CXX=icpx
RUN source /opt/intel/oneapi/setvars.sh && cmake \
  .. \
  -DDNNL_CPU_RUNTIME=OMP \
  -DDNNL_GPU_RUNTIME=NONE \
  -DONEDNN_CPU_RUNTIME=OMP 

# Using all jobs kills my system, so limit to half
# the CPUs
RUN source /opt/intel/oneapi/setvars.sh && make -j$(($(nproc)/2))
RUN source /opt/intel/oneapi/setvars.sh && cmake --build . --target install

# Install Level Zero
#
# In order for sycl-ls to be able to find the Level Zero GPU,
# the container needs to to have --device /dev/dri:/dev/dri
#
# In addition, intel-level-zero-gpu and level-zero packages need to be installed.
#
# Installing from GPU lts/2350 stream
#
# https://dgpu-docs.intel.com/driver/installation.html#ubuntu-install-steps
#   
 
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy/lts/2350 unified" | \
    tee /etc/apt/sources.list.d/intel-gpu-jammy.list

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    level-zero \
    intel-level-zero-gpu \
    intel-opencl-icd \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log}

ENTRYPOINT ["/bin/bash", "-c", "source /opt/intel/oneapi/setvars.sh && /bin/bash"]
