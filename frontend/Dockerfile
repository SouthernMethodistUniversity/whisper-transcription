FROM nginx:alpine

# Copy the frontend files to the Nginx HTML directory
COPY . /usr/share/nginx/html

# Copy custom Nginx configuration
COPY default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]