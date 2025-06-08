from gpiozero import DistanceSensor  # Import the DistanceSensor class from the gpiozero library

import tkinter as tk  # Import the tkinter library for creating the GUI

from tkinter import font  # Import the font module from tkinter for customizing the font

from time import sleep  # Import the sleep function from the time module for delay

# Initialize the ultrasonic sensor

distance_sensor = DistanceSensor(echo=24, trigger=23, max_distance=5)

# Initialize the Tkinter window

window = tk.Tk()

window.title("Distance Measurement")

custom_font = font.Font(size=30)  # Create a custom font object with size 30

window.geometry("800x400")  # Set the dimensions of the window

distance_label = tk.Label(window, text="Distance: ", anchor='center', font=custom_font)
#distance_message = tk.Label(window,test=)

# Create a label to display the distance, centered text, and use the custom font

distance_label.pack()  # Add the label to the window
distance = 600

def measure_distance():

   distance = int(distance_sensor.distance * 100)  # Measure the distance and convert it to an integer

   #distance_label.config(text="Distance: {} cm".format(distance))  # Update the distance label with the new distance

   

   if distance < 20:

       distance_label.config(fg="red", text="Distance: {} cm\nHi!".format(distance))

       # If the distance is less than 20, set the label text to display "Hi!" in red

   elif distance > 30:

       distance_label.config(fg="blue", text="Distance: {} cm\nBye!".format(distance))

       # If the distance is greater than 30, set the label text to display "Bye!" in blue.

       

   window.after(100, measure_distance)  # Schedule the next measurement after 1 second
   

# Start measuring distance

#measure_distance()

# Run the Tkinter event loop

#window.mainloop()

