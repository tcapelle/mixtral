FROM winglian/axolotl:main-py3.10-cu118-2.1.1

WORKDIR /root/src
# RUN mkdir -p /root/src
COPY . /root/src/

# Entry Point
ENTRYPOINT ["accelerate", "launch", "axolotl_launcher.py", "mixtral_axolotl.yml"]