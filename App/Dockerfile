FROM python:3.6
WORKDIR /app
COPY . /app
EXPOSE 5000
RUN pip install --upgrade pip &&  pip install --no-cache-dir -r requirements.txt
CMD ["python","app.py"]