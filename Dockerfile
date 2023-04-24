FROM runpod/pytorch:3.10-2.0.0-117

WORKDIR /

RUN apt-get update && apt-get install -y git wget

# RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh \
#  && chmod +x ~/miniconda.sh \
#  && ~/miniconda.sh -b -p ~/miniconda \
#  && rm ~/miniconda.sh
# ENV PATH=/root/miniconda/bin:$PATH
# ENV CONDA_AUTO_UPDATE_CONDA=false

# # Create a Python 3.6 environment
# RUN /root/miniconda/bin/conda create -y --name py310 python=3.10 \
#  && /root/miniconda/bin/conda clean -ya
# ENV CONDA_DEFAULT_ENV=py310
# ENV CONDA_PREFIX=/root/miniconda/envs/$CONDA_DEFAULT_ENV
# ENV PATH=$CONDA_PREFIX/bin:$PATH

# RUN /root/miniconda/bin/conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia && /root/miniconda/bin/conda clean -ya

RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN python3 -m pip install --upgrade tensorrt
RUN python3 -m pip install --upgrade polygraphy onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
RUN python3 -m pip install onnxruntime

ADD server.py .

RUN git clone https://github.com/FluttyProger/diffusers

WORKDIR /diffusers

RUN pip install .

WORKDIR /

ARG MODEL_NAME
ENV MODEL_NAME=Jeroenvv1985/URPM13_Diffusers

ARG MODEL_REV
ENV MODEL_REV=main

ADD app.py .

# RUN apt-get install --upgrade tensorrt

#RUN wget -O /root/miniconda/envs/py310/lib/python3.10/site-packages/torch/onnx/_constants.py https://raw.githubusercontent.com/pytorch/pytorch/d06d195bcd960f530f8f0d5a1992ed68d2823d4e/torch/onnx/_constants.py
RUN wget -O /usr/local/lib/python3.10/dist-packages/torch/onnx/_constants.py https://raw.githubusercontent.com/pytorch/pytorch/d06d195bcd960f530f8f0d5a1992ed68d2823d4e/torch/onnx/_constants.py

#RUN wget -O /root/miniconda/envs/py310/lib/python3.10/site-packages/torch/onnx/symbolic_opset14.py https://raw.githubusercontent.com/pytorch/pytorch/d06d195bcd960f530f8f0d5a1992ed68d2823d4e/torch/onnx/symbolic_opset14.py
RUN wget -O /usr/local/lib/python3.10/dist-packages/torch/onnx/symbolic_opset14.py https://raw.githubusercontent.com/pytorch/pytorch/d06d195bcd960f530f8f0d5a1992ed68d2823d4e/torch/onnx/symbolic_opset14.py

ADD download.py .
RUN python3 download.py

EXPOSE 8000

CMD python3 -u server.py
