import streamlit as st

def healthcheck():
    return "OK"

if __name__ == "__main__":
    st.write(healthcheck())