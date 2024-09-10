FROM python:3

ENV PYTHONUNBUFFERED True

EXPOSE 8080

ENV APP_HOME /app

WORKDIR $APP_HOME

COPY . ./

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD streamlit run app.py --server.port 8080 --server.enableCORS true --theme.base dark --theme.primaryColor #9b6644 --theme.backgroundColor #0a0a0a --theme.secondaryBackgroundColor #090000 --theme.textColor #f6ebec --theme.font serif