
import requests
import os
import servo
import ultrasonicsensor as us
import cv2
import capturecam as cam
import time

def post_video():
    # Replace 'your_server_ip' with the actual IP address of your server
    server_url = '' # http://xxx.xxx.xx.xx:5000 your server IPv4 address

    # Replace 'image_path' with the path to your image file on your client machine
    file_path = "./videos/output.mp4"
    if not os.path.isfile(file_path):
        print("Error: File not found.")
        return

    filename = os.path.basename(file_path)
    file_extension = os.path.splitext(filename)[1].lower()
    valid_extensions = ('.jpg', '.jpeg', '.png', '.mp4')

    # Open the image file in binary mode
    with open(file_path, "rb") as image_file:

        if file_extension == '.jpg' or file_extension == '.jpeg':
            content_type = "image/jpeg"
        elif file_extension == '.png':
            content_type = "image/png"
        elif file_extension == '.mp4':
            content_type = "video/mp4"
        else:
            content_type = "application/octet-stream"  # Default for unknown types

        files = {"file": (filename, image_file, content_type)}  # Specify content type

        # Send a POST request to the server's upload endpoint
        response = requests.post(server_url, files=files)

    if response.status_code == 200:
        print("Image uploaded successfully!")
        return True
    else:
        print(f"Upload failed: {response.text}")
        return False

if __name__ == "__main__" :
    servo.sg90_close()
    time.sleep(3)
    while True :
        distance = int(us.distance_sensor.distance*100) # distance in cm
        if distance < 20 : 
            print("[ULTRASONICS SENSOR] IN AREA")
            cam.record_video()
            if post_video():
                servo.sg90_open()
                while int(us.distance_sensor.distance*100) < 30 : pass
                time.sleep(3)
                servo.sg90_close()
            
