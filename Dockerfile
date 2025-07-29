FROM python:3.11-slim

WORKDIR /app

# Install dependencies + set Malaysia timezone
RUN apt-get update && \
    apt-get install -y libgomp1 default-mysql-client tzdata && \
    ln -snf /usr/share/zoneinfo/Asia/Kuala_Lumpur /etc/localtime && \
    echo "Asia/Kuala_Lumpur" > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
