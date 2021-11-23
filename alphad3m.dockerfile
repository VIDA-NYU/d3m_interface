FROM registry.gitlab.com/vida-nyu/d3m/alphad3m:latest

# Install d3m_interface
COPY d3m_interface /src/d3m_interface/d3m_interface
COPY MANIFEST.in README.md setup.py /src/d3m_interface/
RUN pip3 freeze | sort >prev_reqs.txt && \
    pip3 install -e /src/d3m_interface && \
    pip3 install notebook && \
    pip3 freeze | sort >new_reqs.txt && \
    comm -23 prev_reqs.txt new_reqs.txt | while read i; do echo "Removed package $i" >&2; exit 1; done && \
    rm prev_reqs.txt new_reqs.txt

ENTRYPOINT ["sh", "-c", "python3 -m ipykernel_launcher --ip=0.0.0.0 -f $1", "--"]
