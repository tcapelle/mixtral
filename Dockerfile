FROM winglian/axolotl:main-latest

WORKDIR /root/src
# RUN mkdir -p /root/src
COPY . /root/src/

# Entry Point
ENTRYPOINT ["accelerate", "launch", "axolotl_launcher.py", "mixtral_axolotl.yml"]