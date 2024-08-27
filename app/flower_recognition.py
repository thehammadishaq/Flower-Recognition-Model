import os
from predict import predict_flower
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_user_input():
    print("Enter the path to the image or type the name of the flower (daisy, dandelion, rose, sunflower, tulip):")
    user_input = input().strip().lower()
    return user_input

def display_image(img_path):
    img = mpimg.imread(img_path)
    imgplot = plt.imshow(img)
    plt.title("Flower Image")
    plt.axis('off')  # Hide axes for better visualization
    plt.show()

def main():
    user_input = get_user_input()

    # Check if the input is an image path
    if os.path.isfile(user_input):
        print("Recognizing flower from the image...")
        result = predict_flower(user_input)
        display_image(user_input)  # Display the image
    else:
        # Assume the input is a flower name
        flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        if user_input in flower_names:
            print(f"You entered a flower name: {user_input.capitalize()}")
            sample_image_path = f'sample_images/{user_input}.jpg'
            if os.path.isfile(sample_image_path):
                display_image(sample_image_path)  # Display sample image for the flower name
                result = user_input.capitalize()
            else:
                print(f"No sample image found for {user_input.capitalize()}")
                result = "No image available"
        else:
            print("Invalid input. Please enter a valid flower name or path to an image.")
            return

    print("Recognition Result:", result)

if __name__ == "__main__":
    main()
