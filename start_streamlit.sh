#!/bin/bash

cd /home/quark/Work/personal-knowledge-base
eval "$(/home/quark/.local/share/reflex/bun/bin/direnv export bash)"
/home/quark/.local/bin/uv run streamlit run streamlit_app.py --server.port $STREAMLIT_APP_PORT
