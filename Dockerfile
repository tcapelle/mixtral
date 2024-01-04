FROM winglian/axolotl:main-py3.10-cu118-2.1.1

# # let's bring in mixtral fixes
# RUN git fetch origin transformers-update-mixtral
# RUN git checkout transformers-update-mixtral
# RUN pip install -e .[deepspeed,flash-attn]

WORKDIR /root/src
# RUN mkdir -p /root/src
COPY . /root/src/

# # install from github repo
# RUN python -m pip install git+https://github.com/huggingface/transformers.git@main

# Entry Point
ENTRYPOINT ["accelerate", "launch", "axolotl_launcher.py", "mixtral_axolotl.yml"]