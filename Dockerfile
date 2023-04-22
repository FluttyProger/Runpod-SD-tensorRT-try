FROM runpod/pytorch:3.10-2.0.0-117

WORKDIR /

RUN apt-get update && apt-get install -y git wget

RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh

RUN chmod -v +x Anaconda*.sh

RUN bash Anaconda3-2023.03-Linux-x86_64.sh -b

RUN rm Anaconda3-2023.03-Linux-x86_64.sh

ENV PATH='$HOME/anaconda3/bin:$PATH'

RUN /root/anaconda3/bin/conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

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

ADD download.py .
RUN python3 download.py

ADD app.py .

RUN wget -O /opt/conda/lib/python3.10/site-packages/torch/onnx/_constants.py https://raw.githubusercontent.com/pytorch/pytorch/d06d195bcd960f530f8f0d5a1992ed68d2823d4e/torch/onnx/_constants.py

RUN wget -O /opt/conda/lib/python3.10/site-packages/torch/onnx/symbolic_opset14.py https://raw.githubusercontent.com/pytorch/pytorch/d06d195bcd960f530f8f0d5a1992ed68d2823d4e/torch/onnx/symbolic_opset14.py

EXPOSE 8000

CMD python3 -u server.py
